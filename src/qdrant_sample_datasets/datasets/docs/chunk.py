import re
import uuid
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from qdrant_sample_datasets.core.records import read_jsonl, write_jsonl
from qdrant_sample_datasets.datasets.docs.content import FENCE_RE, Page, is_fence

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
HEADING_ID_RE = re.compile(r"\s*\{#[^}]*\}")
# Chunks with fewer words than this are merged into a neighbor.
MIN_CHUNK_WORDS = 20

TokenCounter = Callable[[str], int]
Heading = tuple[int, str]
HeadingPath = tuple[Heading, ...]
PieceSizer = Callable[[HeadingPath, str], int]


@dataclass(frozen=True)
class Chunk:
    id: str
    url: str
    title: str
    section: str
    content_type: str
    text: str
    has_code: bool
    dense_text: str
    tokens: int

    @property
    def payload(self) -> dict[str, str | bool]:
        return {
            "url": self.url,
            "title": self.title,
            "section": self.section,
            "content_type": self.content_type,
            "text": self.text,
            "has_code": self.has_code,
        }


@dataclass
class Section:
    heading: Heading | None
    path: HeadingPath
    blocks: list[str] = field(default_factory=list)
    children: list["Section"] = field(default_factory=list)

    @property
    def level(self) -> int:
        return self.heading[0] if self.heading else 1


@dataclass(frozen=True)
class Piece:
    path: HeadingPath
    body: str


def chunk_page(
    page: Page, count_tokens: TokenCounter, budget: int, ceiling: int
) -> list[Chunk]:
    def size(path: HeadingPath, body: str) -> int:
        return count_tokens(dense_text(page, titles(path), body))

    root = build_tree(normalize_headings(list(iter_blocks(page.body))))
    pieces = absorb_tiny(split_section(root, size, budget), size, ceiling)
    return [
        make_chunk(page, piece, index, count_tokens)
        for index, piece in enumerate(pieces)
    ]


def make_chunk(
    page: Page, piece: Piece, index: int, count_tokens: TokenCounter
) -> Chunk:
    stored = chunk_text(page, titles(piece.path), piece.body)
    dense = prose_only(stored)
    return Chunk(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{page.url}#{index}")),
        url=page.url,
        title=page.title,
        section=" > ".join(titles(piece.path)),
        content_type=page.content_type,
        text=stored,
        has_code=FENCE_RE.search(stored) is not None,
        dense_text=dense,
        tokens=count_tokens(dense),
    )


def iter_blocks(body: str) -> Iterator[str]:
    for segment in FENCE_RE.split(body):
        if is_fence(segment):
            yield segment.strip()
            continue
        for paragraph in re.split(r"\n\s*\n", segment):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            yield from split_headings(paragraph)


def split_headings(paragraph: str) -> Iterator[str]:
    lines: list[str] = []
    for line in paragraph.splitlines():
        if HEADING_RE.match(line):
            if lines:
                yield "\n".join(lines)
                lines = []
            yield line
        else:
            lines.append(line)
    if lines:
        yield "\n".join(lines)


def parse_heading(block: str) -> Heading | None:
    match = None if is_fence(block) else HEADING_RE.match(block)
    if match is None:
        return None
    return len(match.group(1)), strip_inline_markup(match.group(2))


def normalize_headings(blocks: list[str]) -> list[str | Heading]:
    """Drop the title H1 and shift levels down when H1 is also used for sections."""
    headings = [parse_heading(block) for block in blocks]
    shift = 1 if sum(1 for h in headings if h and h[0] == 1) > 1 else 0
    items: list[str | Heading] = []
    title_dropped = False
    for block, heading in zip(blocks, headings, strict=True):
        if heading is None:
            items.append(block)
        elif heading[0] == 1 and not title_dropped:
            title_dropped = True
        else:
            items.append((min(heading[0] + shift, 6), heading[1]))
    return items


def build_tree(items: list[str | Heading]) -> Section:
    root = Section(heading=None, path=())
    stack = [root]
    for item in items:
        if isinstance(item, str):
            stack[-1].blocks.append(item)
            continue
        while len(stack) > 1 and stack[-1].level >= item[0]:
            stack.pop()
        node = Section(heading=item, path=(*stack[-1].path, item))
        stack[-1].children.append(node)
        stack.append(node)
    return root


def render(section: Section) -> str:
    parts = list(section.blocks)
    for child in section.children:
        body = render(child)
        if body and child.heading:
            parts += [heading_line(child.heading), body]
    return "\n\n".join(parts)


def split_section(section: Section, size: PieceSizer, budget: int) -> list[Piece]:
    """Keep a section whole when it fits, otherwise split it into its intro and subsections."""
    body = render(section)
    if not body:
        return []
    if size(section.path, body) <= budget:
        return [Piece(section.path, body)]
    pieces = split_intro(section, size, budget)
    for child in section.children:
        pieces += split_section(child, size, budget)
    return pieces


def split_intro(section: Section, size: PieceSizer, budget: int) -> list[Piece]:
    if not section.blocks:
        return []
    texts = pack_blocks(section.blocks, lambda text: size(section.path, text), budget)
    return [Piece(section.path, text) for text in texts]


def absorb_tiny(pieces: list[Piece], size: PieceSizer, ceiling: int) -> list[Piece]:
    backward: list[Piece] = []
    for piece in pieces:
        if backward and is_tiny(piece):
            candidate = combine(backward[-1], piece, size)
            if size(candidate.path, candidate.body) <= ceiling:
                backward[-1] = candidate
                continue
        backward.append(piece)
    forward: list[Piece] = []
    for piece in reversed(backward):
        if forward and is_tiny(piece):
            candidate = combine(piece, forward[-1], size)
            if size(candidate.path, candidate.body) <= ceiling:
                forward[-1] = candidate
                continue
        forward.append(piece)
    return forward[::-1]


def is_tiny(piece: Piece) -> bool:
    return len(piece.body.split()) < MIN_CHUNK_WORDS


def combine(first: Piece, second: Piece, size: PieceSizer) -> Piece:
    """Join two adjacent pieces under the section of the larger one."""
    main = (
        first
        if size(first.path, first.body) >= size(second.path, second.body)
        else second
    )
    parts = [
        piece.body
        if piece is main
        else rebase(piece, len(common_prefix(main.path, piece.path)))
        for piece in (first, second)
    ]
    return Piece(main.path, "\n\n".join(parts))


def rebase(piece: Piece, depth: int) -> str:
    """Return the piece body with the headings below `depth` written into it."""
    return "\n\n".join([*map(heading_line, piece.path[depth:]), piece.body])


def common_prefix(left: HeadingPath, right: HeadingPath) -> HeadingPath:
    prefix: list[Heading] = []
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        prefix.append(a)
    return tuple(prefix)


def heading_line(heading: Heading) -> str:
    return f"{'#' * heading[0]} {heading[1]}"


def titles(path: HeadingPath) -> list[str]:
    return [heading[1] for heading in path]


def pack_blocks(blocks: list[str], size: TokenCounter, budget: int) -> list[str]:
    packed: list[str] = []
    current: list[str] = []
    for block in split_oversized(blocks, size, budget):
        candidate = "\n\n".join(current + [block])
        if current and size(candidate) > budget:
            packed.append("\n\n".join(current))
            current = []
        current.append(block)
    if current:
        packed.append("\n\n".join(current))
    return packed


def split_oversized(
    blocks: list[str], size: TokenCounter, budget: int
) -> Iterator[str]:
    for block in blocks:
        if is_fence(block) or size(block) <= budget:
            yield block
            continue
        lines = block.splitlines()
        header = lines[:2] if is_table(lines) else []
        rows = lines[len(header) :]
        yield from pack_lines(header, rows, size, budget)


def is_table(lines: list[str]) -> bool:
    return (
        len(lines) > 2
        and lines[0].lstrip().startswith("|")
        and set(lines[1].strip()) <= set("|-: ")
    )


def pack_lines(
    header: list[str], rows: list[str], size: TokenCounter, budget: int
) -> Iterator[str]:
    current: list[str] = []
    for row in rows:
        candidate = "\n".join(header + current + [row])
        if current and size(candidate) > budget:
            yield "\n".join(header + current)
            current = []
        current.append(row)
    if current:
        yield "\n".join(header + current)


def chunk_text(page: Page, path: list[str], body: str) -> str:
    heading = " > ".join([page.title, *path])
    return f"{heading}\n\n{body}"


def dense_text(page: Page, path: list[str], body: str) -> str:
    return prose_only(chunk_text(page, path, body))


def prose_only(text: str) -> str:
    text = FENCE_RE.sub("", text)
    text = IMAGE_RE.sub("", text)
    text = LINK_RE.sub(r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text).strip()


def strip_inline_markup(text: str) -> str:
    text = HEADING_ID_RE.sub("", LINK_RE.sub(r"\1", text))
    return text.replace("`", "").replace("*", "").strip()


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    write_jsonl((asdict(chunk) for chunk in chunks), path)


def read_chunks(path: Path) -> list[Chunk]:
    return [Chunk(**row) for row in read_jsonl(path)]
