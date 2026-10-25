from typing import Literal

from qdrant_client import QdrantClient, models

from qdrant_sample_datasets.core.dataset import Hit, Inference
from qdrant_sample_datasets.core.text_models import query_vector
from qdrant_sample_datasets.datasets.docs.model import DENSE_MODEL
from qdrant_sample_datasets.datasets.docs.schema import DENSE_VECTOR, TEXT_FIELD

Mode = Literal["dense", "text", "hybrid", "groups"]
MODES: tuple[Mode, ...] = ("dense", "text", "hybrid", "groups")
PREFETCH_LIMIT = 20


def search(
    client: QdrantClient,
    collection: str,
    query: str,
    mode: Mode,
    inference: Inference,
    limit: int,
) -> list[Hit]:
    text = models.TextQuery(text=query)
    if mode == "text":
        points = client.query_points(
            collection, query=text, using=TEXT_FIELD, limit=limit
        ).points
        return [to_hit(point) for point in points]
    dense = dense_query(query, inference)
    if mode == "dense":
        points = client.query_points(
            collection, query=dense, using=DENSE_VECTOR, limit=limit
        ).points
    elif mode == "hybrid":
        points = client.query_points(
            collection,
            prefetch=prefetches(dense, text),
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        ).points
    else:
        groups = client.query_points_groups(
            collection,
            prefetch=prefetches(dense, text),
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            group_by="url",
            group_size=1,
            limit=limit,
        )
        points = [group.hits[0] for group in groups.groups]
    return [to_hit(point) for point in points]


def dense_query(query: str, inference: Inference) -> list[float] | models.Document:
    return query_vector(DENSE_MODEL, query, inference)


def prefetches(
    dense: list[float] | models.Document, text: models.TextQuery
) -> list[models.Prefetch]:
    return [
        models.Prefetch(query=dense, using=DENSE_VECTOR, limit=PREFETCH_LIMIT),
        models.Prefetch(query=text, using=TEXT_FIELD, limit=PREFETCH_LIMIT),
    ]


def to_hit(point: models.ScoredPoint) -> Hit:
    payload = point.payload or {}
    return Hit(
        score=point.score,
        title=f"[{payload.get('content_type', '')}] {payload.get('title', '')} > {payload.get('section', '')}",
        location=str(payload.get("url", "")),
    )
