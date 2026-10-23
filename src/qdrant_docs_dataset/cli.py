import argparse
import os
import statistics
from collections import Counter
from pathlib import Path

from qdrant_client import QdrantClient

from qdrant_docs_dataset.build import build_collection, download_snapshot
from qdrant_docs_dataset.chunk import Chunk, chunk_page, read_chunks, write_chunks
from qdrant_docs_dataset.content import CONTENT_TYPES, load_pages
from qdrant_docs_dataset.embed import TOKEN_BUDGET, count_tokens
from qdrant_docs_dataset.search import search

DEFAULT_COLLECTION = "qdrant-docs"


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdrant-docs-dataset")
    commands = parser.add_subparsers(required=True)

    chunk = commands.add_parser(
        "chunk", help="Turn landing page content into chunks.jsonl"
    )
    chunk.add_argument(
        "--content", type=Path, required=True, help="Path to qdrant-landing/content"
    )
    chunk.add_argument("--out", type=Path, default=Path("chunks.jsonl"))
    chunk.add_argument(
        "--sections",
        nargs="+",
        default=list(CONTENT_TYPES),
        choices=list(CONTENT_TYPES),
    )
    chunk.add_argument("--budget", type=int, default=TOKEN_BUDGET)
    chunk.set_defaults(func=run_chunk)

    for name, func, help_text in (
        ("build", run_build, "Create the collection and upsert chunks"),
        ("query", run_query, "Run a search against the collection"),
        ("snapshot", run_snapshot, "Create and download a collection snapshot"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument(
            "--url", default=os.environ.get("QDRANT_URL", "http://localhost:6333")
        )
        command.add_argument("--api-key", default=os.environ.get("QDRANT_API_KEY"))
        command.add_argument("--collection", default=DEFAULT_COLLECTION)
        command.add_argument("--inference", choices=["local", "cloud"], default="local")
        command.set_defaults(func=func)

    commands.choices["build"].add_argument(
        "--chunks", type=Path, default=Path("chunks.jsonl")
    )
    commands.choices["query"].add_argument("text")
    commands.choices["query"].add_argument(
        "--mode", choices=["dense", "text", "hybrid", "groups"], default="hybrid"
    )
    commands.choices["query"].add_argument("--limit", type=int, default=5)
    commands.choices["snapshot"].add_argument(
        "--out", type=Path, default=Path("snapshots")
    )
    return parser


def run_chunk(args: argparse.Namespace) -> None:
    pages = load_pages(args.content, args.sections)
    chunks: list[Chunk] = []
    for page in pages:
        chunks.extend(chunk_page(page, count_tokens, args.budget))
    write_chunks(chunks, args.out)
    report(pages_count=len(pages), chunks=chunks, budget=args.budget)


def report(pages_count: int, chunks: list[Chunk], budget: int) -> None:
    tokens = sorted(chunk.tokens for chunk in chunks)
    by_type = Counter(chunk.content_type for chunk in chunks)
    print(f"pages: {pages_count}  chunks: {len(chunks)}  {dict(by_type)}")
    print(
        f"tokens p50={statistics.median(tokens):.0f} p90={tokens[int(len(tokens) * 0.9)]} "
        f"max={tokens[-1]} over_budget={sum(t > budget for t in tokens)}"
    )


def client_from(args: argparse.Namespace) -> QdrantClient:
    return QdrantClient(url=args.url, api_key=args.api_key, timeout=120)


def run_build(args: argparse.Namespace) -> None:
    chunks = read_chunks(args.chunks)
    client = client_from(args)
    build_collection(client, args.collection, chunks, args.inference)
    print(f"collection {args.collection}: {client.count(args.collection).count} points")


def run_query(args: argparse.Namespace) -> None:
    hits = search(
        client_from(args),
        args.collection,
        args.text,
        args.mode,
        args.inference,
        args.limit,
    )
    for hit in hits:
        location = f"{hit.url}#{hit.anchor}" if hit.anchor else hit.url
        print(
            f"{hit.score:8.4f}  [{hit.content_type}] {hit.title} > {hit.section}\n          {location}"
        )


def run_snapshot(args: argparse.Namespace) -> None:
    target = download_snapshot(
        client_from(args), args.url, args.api_key, args.collection, args.out
    )
    print(f"snapshot written to {target} ({target.stat().st_size / 1e6:.1f} MB)")
