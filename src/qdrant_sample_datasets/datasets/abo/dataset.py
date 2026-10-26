from argparse import ArgumentParser, Namespace
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

from qdrant_client import QdrantClient, models

from qdrant_sample_datasets.core.dataset import Dataset, Hit, Inference
from qdrant_sample_datasets.core.records import batched, read_jsonl, write_jsonl
from qdrant_sample_datasets.datasets.abo.schema import (
    COLLECTION,
    IMAGE_MODEL,
)
from qdrant_sample_datasets.datasets.abo.search import (
    MODES,
    search_by_image,
    search_by_text,
)
from qdrant_sample_datasets.datasets.abo.source import (
    PHONE_CASE_TYPE,
    Product,
    download,
    load,
)

BATCH_SIZE = 64
RAW_DIR = Path("data/abo/raw")
CLOUD_ONLY = (
    "abo uses the CLIP models of Qdrant Cloud Inference. Run it with --inference cloud."
)


class AboDataset(Dataset):
    name = "abo"
    description = (
        "Amazon Berkeley Objects products, searchable by text and by image with CLIP"
    )
    collection = COLLECTION

    def add_prepare_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--raw-dir",
            type=Path,
            default=RAW_DIR,
            help="Where the source files are kept",
        )
        parser.add_argument(
            "--phone-case-cap",
            type=int,
            default=1000,
            help="Phone cases to keep, chosen at random",
        )
        parser.add_argument("--seed", type=int, default=0)

    def add_query_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--mode",
            choices=MODES,
            default="hybrid",
            help="How to search with a text query",
        )
        parser.add_argument(
            "--image", action="store_true", help="Treat the query as an image URL"
        )

    def prepare(self, args: Namespace, out: Path) -> None:
        download(args.raw_dir)
        products = load(args.raw_dir, args.phone_case_cap, args.seed)
        write_jsonl((asdict(product) for product in products), out)
        report(products)

    def points(
        self, records: Path, inference: Inference
    ) -> Iterator[list[models.PointStruct]]:
        if inference != "cloud":
            raise SystemExit(CLOUD_ONLY)
        products = [Product(**row) for row in read_jsonl(records)]
        for batch in batched(products, BATCH_SIZE):
            yield [
                models.PointStruct(
                    id=product.id,
                    vector=models.Image(image=product.image_url, model=IMAGE_MODEL),
                    payload=product.payload,
                )
                for product in batch
            ]

    def search(
        self,
        client: QdrantClient,
        collection: str,
        text: str,
        args: Namespace,
        inference: Inference,
    ) -> list[Hit]:
        if inference != "cloud" and (args.image or args.mode != "text"):
            raise SystemExit(CLOUD_ONLY)
        if args.image:
            points = search_by_image(client, collection, text, args.limit)
        else:
            points = search_by_text(client, collection, text, args.mode, args.limit)
        return [to_hit(point) for point in points]


def to_hit(point: models.ScoredPoint) -> Hit:
    payload = point.payload or {}
    return Hit(
        score=point.score,
        title=f"{payload.get('name', '')} [{payload.get('product_type', '')}]",
        location=str(payload.get("image_url", "")),
    )


def report(products: list[Product]) -> None:
    types = Counter(product.product_type for product in products)
    missing = {
        field: sum(1 for p in products if not getattr(p, field))
        for field in ("brand", "color", "product_type")
    }
    print(
        f"products: {len(products)}  product types: {len(types)}  phone cases: {types[PHONE_CASE_TYPE]}"
    )
    print(f"missing: {missing}")
    print(f"top types: {types.most_common(8)}")
