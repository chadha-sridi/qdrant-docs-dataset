from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from qdrant_client import QdrantClient, models

Inference = Literal["local", "cloud"]


@dataclass(frozen=True)
class CollectionSpec:
    default_name: str
    # One unnamed vector, or several named vectors.
    vectors: models.VectorParams | dict[str, models.VectorParams]
    payload_indexes: dict[str, models.PayloadFieldSchema]
    # Lower than Qdrant's default for small collections, so the HNSW graph gets built.
    indexing_threshold_kb: int | None = None


@dataclass(frozen=True)
class Hit:
    score: float
    title: str
    location: str


class Dataset(ABC):
    """A sample dataset: how to prepare its records, store them, and query them."""

    name: str
    description: str
    collection: CollectionSpec

    def add_prepare_arguments(self, parser: ArgumentParser) -> None:
        """Add dataset-specific options to the prepare step."""

    def add_query_arguments(self, parser: ArgumentParser) -> None:
        """Add dataset-specific options to the query step."""

    @abstractmethod
    def prepare(self, args: Namespace, out: Path) -> None:
        """Download and clean the source data, then write records to `out` as JSONL."""

    @abstractmethod
    def points(
        self, records: Path, inference: Inference
    ) -> Iterator[list[models.PointStruct]]:
        """Read prepared records and yield batches of points to upsert."""

    supports_evaluation = False

    def add_evaluate_arguments(self, parser: ArgumentParser) -> None:
        """Add dataset-specific options to the evaluate step."""

    def evaluate(
        self,
        client: QdrantClient,
        collection: str,
        args: Namespace,
        inference: Inference,
    ) -> None:
        """Score the collection against the dataset's relevance judgments, if it ships any."""
        raise NotImplementedError(f"{self.name} has no relevance judgments")

    @abstractmethod
    def search(
        self,
        client: QdrantClient,
        collection: str,
        text: str,
        args: Namespace,
        inference: Inference,
    ) -> list[Hit]:
        """Run a query against the collection."""
