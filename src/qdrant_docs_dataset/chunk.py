import json
import re
import uuid
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from qdrant_docs_dataset.content import FENCE_RE, Page, is_fence

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class Chunk:
    id: str
    url: str
    title: str
    section: str
    content_type: str
    text: str
    dense_text: str
    tokens: int

    @property
    def payload(self) -> dict[str, str]:
        return {
            "url": self.url,
            "title": self.title,
            "section": self.section,
            "content_type": self.content_type,
            "text": self.text,
        }


@dataclass
class Leaf:
    path: list[str]
    blocks: list[str]

    @property
    def text(self) -> str:
        return "\n\n".join(self.blocks)


def chunk_page(page: Page, count_tokens: TokenCounter, budget: int) -> list[Chunk]:
    leaves = list(split_leaves(page.body))
    groups = merge_leaves(
        leaves,
        lambda leaf: count_tokens(dense_text(page, leaf.path, leaf.text)),
        budget,
    )
    chunks: list[Chunk] = []
    for group in groups:
        for text in pack_blocks(
            group.blocks, group_size_counter(page, group, count_tokens), budget
        ):
            chunks.append(make_chunk(page, group, text, len(chunks), count_tokens))
    return chunks


def group_size_counter(
    page: Page, group: Leaf, count_tokens: TokenCounter
) -> TokenCounter:
    return lambda text: count_tokens(dense_text(page, group.path, text))


def make_chunk(
    page: Page, group: Leaf, text: str, index: int, count_tokens: TokenCounter
) -> Chunk:
    stored = chunk_text(page, group.path, text)
    dense = prose_only(stored)
    return Chunk(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{page.url}#{index}")),
        url=page.url,
        title=page.title,
        section=" > ".join(group.path),
        content_type=page.content_type,
        text=stored,
        dense_text=dense,
        tokens=count_tokens(dense),
    )


def split_leaves(body: str) -> Iterator[Leaf]:
    leaf = Leaf(path=[], blocks=[])
    for block in iter_blocks(body):
        heading = None if is_fence(block) else HEADING_RE.match(block)
        if heading is None or len(heading.group(1)) == 1:
            if heading is None:
                leaf.blocks.append(block)
            continue
        if leaf.blocks:
            yield leaf
        level, text = len(heading.group(1)), strip_inline_markup(heading.group(2))
        path = leaf.path[: level - 2] + [text]
        leaf = Leaf(path=path, blocks=[])
    if leaf.blocks:
        yield leaf


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


def merge_leaves(
    leaves: list[Leaf], size: Callable[[Leaf], int], budget: int
) -> list[Leaf]:
    merged: list[Leaf] = []
    for leaf in leaves:
        if merged and same_top_section(merged[-1], leaf):
            candidate = Leaf(
                path=common_prefix(merged[-1].path, leaf.path),
                blocks=merged[-1].blocks + leaf.blocks,
            )
            if size(candidate) <= budget:
                merged[-1] = candidate
                continue
        merged.append(leaf)
    return merged


def same_top_section(left: Leaf, right: Leaf) -> bool:
    return bool(left.path) and bool(right.path) and left.path[0] == right.path[0]


def common_prefix(left: list[str], right: list[str]) -> list[str]:
    prefix: list[str] = []
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        prefix.append(a)
    return prefix


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
    return LINK_RE.sub(r"\1", text).replace("`", "").replace("*", "").strip()


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as handle:
        return [Chunk(**json.loads(line)) for line in handle if line.strip()]
