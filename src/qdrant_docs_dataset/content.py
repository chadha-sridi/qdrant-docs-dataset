import html
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

CONTENT_TYPES = {
    "documentation": "documentation",
    "articles": "article",
    "course": "course",
}
EXCLUDED_PATH_PARTS = {"headless", "snippets"}
EXCLUDED_PAGE_TYPES = {"delimiter", "external-link"}
MIN_BODY_WORDS = 40
# Snippet files to prefer, in order. Other languages are used only when none of these exist.
SNIPPET_LANGUAGES = ("python", "http", "bash", "shell")
SNIPPET_DIRS = ("generated", "_server/generated", "", "_server")
# SDK languages that duplicate a Python block placed next to them.
ALTERNATIVE_LANGUAGES = {
    "typescript",
    "ts",
    "javascript",
    "js",
    "rust",
    "java",
    "go",
    "csharp",
    "cs",
    "http",
}
PYTHON_LANGUAGES = {"python", "py"}

FRONTMATTER_RE = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)
CODE_SNIPPET_RE = re.compile(r"\{\{<\s*code-snippet\s+([^>]*?)\s*>\}\}")
ATTRIBUTE_RE = re.compile(r'(\w+)="([^"]*)"')
FENCE_LANGUAGE_RE = re.compile(r"[ \t]*`{3,}[ \t]*([^\s`{]*)")
SHORTCODE_RE = re.compile(r"\{\{[<%].*?[>%]\}\}", re.DOTALL)
FENCE_RE = re.compile(
    r"(^[ \t]*`{3,}[^`\n]*\n.*?^[ \t]*`{3,}[ \t]*$)", re.DOTALL | re.MULTILINE
)
URL_UNSAFE_RE = re.compile(r"[^\w.-]")
INLINE_CODE_RE = re.compile(r"(`[^`\n]+`)")
BLOCK_HTML_RE = re.compile(
    r"<(svg|script|style|iframe|video)\b.*?</\1>", re.DOTALL | re.IGNORECASE
)
TAG_RE = re.compile(r"</?[a-zA-Z][^<>]*>")
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


@dataclass(frozen=True)
class Page:
    url: str
    title: str
    content_type: str
    body: str
    source_path: str


def load_pages(content_root: Path, sections: list[str]) -> list[Page]:
    pages: list[Page] = []
    for section in sections:
        for path in sorted((content_root / section).rglob("*.md")):
            page = load_page(content_root, section, path)
            if page is not None:
                pages.append(page)
    return pages


def load_page(content_root: Path, section: str, path: Path) -> Page | None:
    relative = path.relative_to(content_root)
    if EXCLUDED_PATH_PARTS & set(relative.parts):
        return None
    meta, raw_body = split_frontmatter(path.read_text(encoding="utf-8-sig"))
    if meta is None or is_excluded(meta):
        return None
    body = clean_body(content_root, raw_body)
    if len(body.split()) < MIN_BODY_WORDS:
        return None
    title = str(meta.get("title") or relative.stem.replace("-", " ").title())
    return Page(
        url=page_url(relative, meta),
        title=title,
        content_type=CONTENT_TYPES[section],
        body=body,
        source_path=str(relative),
    )


def split_frontmatter(text: str) -> tuple[dict | None, str]:
    match = FRONTMATTER_RE.match(text)
    if match is None:
        return (None, "") if text.startswith("---") else ({}, text)
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return None, ""
    if not isinstance(meta, dict):
        return None, ""
    return meta, text[match.end() :]


def is_excluded(meta: dict) -> bool:
    if meta.get("draft") is True:
        return True
    if meta.get("type") in EXCLUDED_PAGE_TYPES:
        return True
    return bool(meta.get("sitemapExclude"))


def page_url(relative: Path, meta: dict) -> str:
    if meta.get("url"):
        return "/" + str(meta["url"]).strip("/") + "/"
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "_index":
        parts.pop()
    elif meta.get("slug"):
        parts[-1] = str(meta["slug"])
    return "/" + "/".join(URL_UNSAFE_RE.sub("", part.lower()) for part in parts) + "/"


def clean_body(content_root: Path, body: str) -> str:
    body = CODE_SNIPPET_RE.sub(
        lambda m: load_snippet(content_root, dict(ATTRIBUTE_RE.findall(m.group(1)))),
        body,
    )
    body = drop_python_alternatives(body)
    segments = FENCE_RE.split(body)
    cleaned = [seg if is_fence(seg) else clean_prose(seg) for seg in segments]
    return re.sub(r"\n{3,}", "\n\n", "".join(cleaned)).strip()


def is_fence(segment: str) -> bool:
    return segment.lstrip(" ").startswith("```")


def load_snippet(content_root: Path, attributes: dict[str, str]) -> str:
    base = content_root / attributes["path"].strip("/")
    block = attributes.get("block")
    dirs = [base / "generated" / block] if block else [base / d for d in SNIPPET_DIRS]
    candidates = [d / f"{language}.md" for language in SNIPPET_LANGUAGES for d in dirs]
    candidates += [
        f
        for d in dirs
        if d.is_dir()
        for f in sorted(d.glob("*.md"))
        if not f.name.startswith("_")
    ]
    for candidate in candidates:
        if candidate.is_file():
            code = candidate.read_text(encoding="utf-8").strip()
            return f"\n{code}\n" if code.startswith("```") else f"\n```\n{code}\n```\n"
    return "\n"


def drop_python_alternatives(body: str) -> str:
    segments = FENCE_RE.split(body)
    for group in fence_groups(segments):
        languages = [fence_language(segments[i]) for i in group]
        if PYTHON_LANGUAGES & set(languages):
            for index, language in zip(group, languages, strict=True):
                if language in ALTERNATIVE_LANGUAGES:
                    segments[index] = ""
    return "".join(segments)


def fence_groups(segments: list[str]) -> list[list[int]]:
    groups: list[list[int]] = []
    for index in range(1, len(segments), 2):
        if groups and not segments[index - 1].strip():
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def fence_language(fence: str) -> str:
    match = FENCE_LANGUAGE_RE.match(fence)
    return match.group(1).lower() if match else ""


def clean_prose(text: str) -> str:
    text = SHORTCODE_RE.sub("", text)
    text = COMMENT_RE.sub("", text)
    text = BLOCK_HTML_RE.sub("", text)
    parts = INLINE_CODE_RE.split(text)
    parts = [
        part if part.startswith("`") else html.unescape(TAG_RE.sub("", part))
        for part in parts
    ]
    return "".join(parts)
