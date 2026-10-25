import statistics
from argparse import ArgumentParser, Namespace
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from qdrant_client import QdrantClient, models

from qdrant_sample_datasets.core.dataset import Dataset, Hit, Inference
from qdrant_sample_datasets.core.records import batched
from qdrant_sample_datasets.core.text_models import passage_vectors
from qdrant_sample_datasets.datasets.docs.chunk import (
    Chunk,
    chunk_page,
    read_chunks,
    write_chunks,
)
from qdrant_sample_datasets.datasets.docs.content import CONTENT_TYPES, load_pages
from qdrant_sample_datasets.datasets.docs.model import (
    DENSE_MODEL,
    TOKEN_BUDGET,
    TOKEN_CEILING,
    count_tokens,
)
from qdrant_sample_datasets.datasets.docs.schema import COLLECTION, DENSE_VECTOR
from qdrant_sample_datasets.datasets.docs.search import MODES, search

BATCH_SIZE = 64


class DocsDataset(Dataset):
    name = "docs"
    description = "Qdrant documentation, articles, and course, chunked by section for hybrid search"
    collection = COLLECTION

    def add_prepare_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--content", type=Path, required=True, help="Path to qdrant-landing/content"
        )
        parser.add_argument(
            "--sections",
            nargs="+",
            default=list(CONTENT_TYPES),
            choices=list(CONTENT_TYPES),
        )
        parser.add_argument("--budget", type=int, default=TOKEN_BUDGET)

    def add_query_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--mode", choices=MODES, default="hybrid")

    def prepare(self, args: Namespace, out: Path) -> None:
        pages = load_pages(args.content, args.sections)
        chunks = [
            chunk
            for page in pages
            for chunk in chunk_page(page, count_tokens, args.budget, TOKEN_CEILING)
        ]
        write_chunks(chunks, out)
        report(len(pages), chunks, args.budget)

    def points(
        self, records: Path, inference: Inference
    ) -> Iterator[list[models.PointStruct]]:
        for batch in batched(read_chunks(records), BATCH_SIZE):
            vectors = passage_vectors(
                DENSE_MODEL, [chunk.dense_text for chunk in batch], inference
            )
            yield [
                models.PointStruct(
                    id=chunk.id, vector={DENSE_VECTOR: vector}, payload=chunk.payload
                )
                for chunk, vector in zip(batch, vectors, strict=True)
            ]

    def search(
        self,
        client: QdrantClient,
        collection: str,
        text: str,
        args: Namespace,
        inference: Inference,
    ) -> list[Hit]:
        return search(client, collection, text, args.mode, inference, args.limit)


def report(pages_count: int, chunks: list[Chunk], budget: int) -> None:
    tokens = sorted(chunk.tokens for chunk in chunks)
    by_type = Counter(chunk.content_type for chunk in chunks)
    print(f"pages: {pages_count}  chunks: {len(chunks)}  {dict(by_type)}")
    print(
        f"tokens p50={statistics.median(tokens):.0f} p90={tokens[int(len(tokens) * 0.9)]} "
        f"max={tokens[-1]} over_budget={sum(t > budget for t in tokens)} "
        f"over_ceiling={sum(t > TOKEN_CEILING for t in tokens)}"
    )
