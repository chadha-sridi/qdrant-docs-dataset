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
SNIPPET_VARIANTS = ("_server/generated/python.md", "_server/http.md", "_server/bash.md")

FRONTMATTER_RE = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)
CODE_SNIPPET_RE = re.compile(r'\{\{<\s*code-snippet\s+path="([^"]+)"\s*>\}\}')
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
    body = CODE_SNIPPET_RE.sub(lambda m: load_snippet(content_root, m.group(1)), body)
    segments = FENCE_RE.split(body)
    cleaned = [seg if is_fence(seg) else clean_prose(seg) for seg in segments]
    return re.sub(r"\n{3,}", "\n\n", "".join(cleaned)).strip()


def is_fence(segment: str) -> bool:
    return segment.lstrip(" ").startswith("```")


def load_snippet(content_root: Path, snippet_path: str) -> str:
    base = content_root / snippet_path.strip("/")
    for variant in SNIPPET_VARIANTS:
        candidate = base / variant
        if candidate.exists():
            code = candidate.read_text(encoding="utf-8").strip()
            return f"\n{code}\n" if code.startswith("```") else f"\n```\n{code}\n```\n"
    return "\n"


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
