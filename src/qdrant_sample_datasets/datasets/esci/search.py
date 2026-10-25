from typing import Literal

from qdrant_client import QdrantClient, models

from qdrant_sample_datasets.core.dataset import Inference
from qdrant_sample_datasets.core.text_models import query_vector
from qdrant_sample_datasets.datasets.esci.schema import (
    DENSE_MODEL,
    DENSE_VECTOR,
    TEXT_FIELD,
)

Mode = Literal["dense", "text", "hybrid"]
MODES: tuple[Mode, ...] = ("dense", "text", "hybrid")
PREFETCH_LIMIT = 50


def search(
    client: QdrantClient,
    collection: str,
    query: str,
    mode: Mode,
    inference: Inference,
    limit: int,
) -> list[models.ScoredPoint]:
    text = models.TextQuery(text=query)
    if mode == "text":
        return client.query_points(
            collection, query=text, using=TEXT_FIELD, limit=limit
        ).points
    dense = query_vector(DENSE_MODEL, query, inference)
    if mode == "dense":
        return client.query_points(
            collection, query=dense, using=DENSE_VECTOR, limit=limit
        ).points
    return client.query_points(
        collection,
        prefetch=[
            models.Prefetch(query=dense, using=DENSE_VECTOR, limit=PREFETCH_LIMIT),
            models.Prefetch(query=text, using=TEXT_FIELD, limit=PREFETCH_LIMIT),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
    ).points
