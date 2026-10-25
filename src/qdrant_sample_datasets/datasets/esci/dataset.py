import statistics
from argparse import ArgumentParser, Namespace
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

from qdrant_client import QdrantClient, models

from qdrant_sample_datasets.core.dataset import Dataset, Hit, Inference
from qdrant_sample_datasets.core.records import batched, read_jsonl, write_jsonl
from qdrant_sample_datasets.core.text_models import passage_vectors, tokenizer
from qdrant_sample_datasets.datasets.esci.evaluate import (
    CUTOFF,
    mean_ndcg,
    sample_queries,
)
from qdrant_sample_datasets.datasets.esci.schema import (
    COLLECTION,
    DENSE_MODEL,
    DENSE_VECTOR,
)
from qdrant_sample_datasets.datasets.esci.search import MODES, Mode, search
from qdrant_sample_datasets.datasets.esci.source import Product, Query, download, load

BATCH_SIZE = 128
RAW_DIR = Path("data/esci/raw")
QUERIES_FILE = "queries.jsonl"


class EsciDataset(Dataset):
    name = "esci"
    description = (
        "Amazon product search (ESCI), US test queries with graded relevance judgments"
    )
    collection = COLLECTION
    supports_evaluation = True

    def add_prepare_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--raw-dir",
            type=Path,
            default=RAW_DIR,
            help="Where the source Parquet files are kept",
        )

    def add_query_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--mode", choices=MODES, default="hybrid")

    def add_evaluate_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--queries", type=Path, default=Path("data/esci") / QUERIES_FILE
        )
        parser.add_argument("--mode", choices=[*MODES, "all"], default="all")
        parser.add_argument(
            "--sample", type=int, help="Evaluate a random sample of this many queries"
        )
        parser.add_argument("--seed", type=int, default=7)

    def prepare(self, args: Namespace, out: Path) -> None:
        download(args.raw_dir)
        products, queries = load(args.raw_dir)
        write_jsonl((asdict(product) for product in products), out)
        write_jsonl((asdict(query) for query in queries), out.with_name(QUERIES_FILE))
        report(products, queries)

    def points(
        self, records: Path, inference: Inference
    ) -> Iterator[list[models.PointStruct]]:
        products = [Product(**row) for row in read_jsonl(records)]
        for batch in batched(products, BATCH_SIZE):
            vectors = passage_vectors(
                DENSE_MODEL, [product.text for product in batch], inference
            )
            yield [
                models.PointStruct(
                    id=product.id,
                    vector={DENSE_VECTOR: vector},
                    payload=product.payload,
                )
                for product, vector in zip(batch, vectors, strict=True)
            ]

    def search(
        self,
        client: QdrantClient,
        collection: str,
        text: str,
        args: Namespace,
        inference: Inference,
    ) -> list[Hit]:
        return [
            to_hit(point)
            for point in search(
                client, collection, text, args.mode, inference, args.limit
            )
        ]

    def evaluate(
        self,
        client: QdrantClient,
        collection: str,
        args: Namespace,
        inference: Inference,
    ) -> None:
        queries = sample_queries(
            [Query(**row) for row in read_jsonl(args.queries)], args.sample, args.seed
        )
        modes: tuple[Mode, ...] = MODES if args.mode == "all" else (args.mode,)
        print(f"nDCG@{CUTOFF} over {len(queries)} queries")
        for mode in modes:

            def ranked_ids(query: Query, mode: Mode = mode) -> list[str]:
                points = search(
                    client, collection, query.query, mode, inference, CUTOFF
                )
                return [str((point.payload or {})["product_id"]) for point in points]

            print(f"  {mode:7} {mean_ndcg(queries, ranked_ids):.3f}")


def to_hit(point: models.ScoredPoint) -> Hit:
    payload = point.payload or {}
    brand = f" | {payload['brand']}" if payload.get("brand") else ""
    return Hit(
        score=point.score,
        title=f"{payload.get('title', '')}{brand}",
        location=str(payload.get("product_id")),
    )


def report(products: list[Product], queries: list[Query]) -> None:
    labels = Counter(label for query in queries for label in query.judgments.values())
    encodings = tokenizer(DENSE_MODEL.name).encode_batch(
        [p.text for p in products], add_special_tokens=False
    )
    tokens = sorted(len(encoding.ids) for encoding in encodings)
    over = sum(t > DENSE_MODEL.context_window - 5 for t in tokens)
    print(
        f"products: {len(products)}  queries: {len(queries)}  judgments: {sum(labels.values())}  {dict(labels)}"
    )
    print(
        f"tokens p50={statistics.median(tokens):.0f} p90={tokens[int(len(tokens) * 0.9)]} max={tokens[-1]} truncated={over}"
    )
