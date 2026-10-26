import argparse
import os
from pathlib import Path

from qdrant_sample_datasets.core.dataset import Dataset
from qdrant_sample_datasets.core.qdrant import (
    build_collection,
    connect,
    download_snapshot,
    limit_points,
)
from qdrant_sample_datasets.datasets import DATASETS

DATA_DIR = Path("data")
SNAPSHOT_DIR = Path("snapshots")


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qdrant-datasets", description="Build Qdrant sample dataset snapshots."
    )
    datasets = parser.add_subparsers(dest="dataset", required=True, metavar="DATASET")
    datasets.add_parser("list", help="List the available datasets").set_defaults(
        func=run_list
    )
    for dataset in DATASETS.values():
        add_dataset_commands(
            datasets.add_parser(dataset.name, help=dataset.description), dataset
        )
    return parser


def add_dataset_commands(parser: argparse.ArgumentParser, dataset: Dataset) -> None:
    steps = parser.add_subparsers(dest="step", required=True, metavar="STEP")
    records = DATA_DIR / dataset.name / "records.jsonl"

    prepare = steps.add_parser(
        "prepare", help="Download and clean the source data into records"
    )
    prepare.add_argument("--out", type=Path, default=records)
    dataset.add_prepare_arguments(prepare)
    prepare.set_defaults(func=lambda args: dataset.prepare(args, args.out))

    build = steps.add_parser(
        "build", help="Create the collection and upsert the records"
    )
    add_connection_arguments(build, dataset)
    build.add_argument("--records", type=Path, default=records)
    build.add_argument(
        "--limit", type=int, help="Upsert only the first N records, for a test build"
    )
    build.set_defaults(func=lambda args: run_build(args, dataset))

    query = steps.add_parser("query", help="Run a search against the collection")
    add_connection_arguments(query, dataset)
    query.add_argument("text")
    query.add_argument("--limit", type=int, default=5)
    dataset.add_query_arguments(query)
    query.set_defaults(func=lambda args: run_query(args, dataset))

    if dataset.supports_evaluation:
        evaluate = steps.add_parser(
            "evaluate", help="Score the collection against the relevance judgments"
        )
        add_connection_arguments(evaluate, dataset)
        dataset.add_evaluate_arguments(evaluate)
        evaluate.set_defaults(func=lambda args: run_evaluate(args, dataset))

    snapshot = steps.add_parser(
        "snapshot", help="Create and download a collection snapshot"
    )
    add_connection_arguments(snapshot, dataset, inference=False)
    snapshot.add_argument("--out", type=Path, default=SNAPSHOT_DIR / dataset.name)
    snapshot.set_defaults(func=run_snapshot)


def add_connection_arguments(
    parser: argparse.ArgumentParser, dataset: Dataset, inference: bool = True
) -> None:
    parser.add_argument(
        "--url", default=os.environ.get("QDRANT_URL", "http://localhost:6333")
    )
    parser.add_argument("--api-key", default=os.environ.get("QDRANT_API_KEY"))
    parser.add_argument("--collection", default=dataset.collection.default_name)
    if inference:
        parser.add_argument("--inference", choices=["local", "cloud"], default="local")


def run_list(args: argparse.Namespace) -> None:
    for dataset in DATASETS.values():
        print(f"{dataset.name:12} {dataset.description}")


def run_build(args: argparse.Namespace, dataset: Dataset) -> None:
    client = connect(args.url, args.api_key, args.inference)
    count = build_collection(
        client,
        args.collection,
        dataset.collection,
        limit_points(dataset.points(args.records, args.inference), args.limit),
    )
    print(f"collection {args.collection}: {count} points")


def run_query(args: argparse.Namespace, dataset: Dataset) -> None:
    client = connect(args.url, args.api_key, args.inference)
    for hit in dataset.search(client, args.collection, args.text, args, args.inference):
        print(f"{hit.score:8.4f}  {hit.title}\n          {hit.location}")


def run_evaluate(args: argparse.Namespace, dataset: Dataset) -> None:
    client = connect(args.url, args.api_key, args.inference)
    dataset.evaluate(client, args.collection, args, args.inference)


def run_snapshot(args: argparse.Namespace) -> None:
    client = connect(args.url, args.api_key)
    target = download_snapshot(
        client, args.url, args.api_key, args.collection, args.out
    )
    print(f"snapshot written to {target} ({target.stat().st_size / 1e6:.1f} MB)")
