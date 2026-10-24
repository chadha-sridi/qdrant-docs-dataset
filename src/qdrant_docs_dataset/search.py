from dataclasses import dataclass
from typing import Literal

from qdrant_client import QdrantClient, models

from qdrant_docs_dataset.build import DENSE_VECTOR, TEXT_FIELD, Inference
from qdrant_docs_dataset.embed import DENSE_MODEL, embed_query

Mode = Literal["dense", "text", "hybrid", "groups"]
PREFETCH_LIMIT = 20


@dataclass(frozen=True)
class Hit:
    score: float
    url: str
    title: str
    section: str
    content_type: str


def search(
    client: QdrantClient,
    collection: str,
    query: str,
    mode: Mode,
    inference: Inference,
    limit: int,
) -> list[Hit]:
    dense = dense_query(query, inference)
    text = models.TextQuery(text=query)
    if mode == "dense":
        response = client.query_points(
            collection, query=dense, using=DENSE_VECTOR, limit=limit
        )
    elif mode == "text":
        response = client.query_points(
            collection, query=text, using=TEXT_FIELD, limit=limit
        )
    elif mode == "hybrid":
        response = client.query_points(
            collection,
            prefetch=prefetches(dense, text),
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )
    else:
        groups = client.query_points_groups(
            collection,
            prefetch=prefetches(dense, text),
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            group_by="url",
            group_size=1,
            limit=limit,
        )
        return [to_hit(group.hits[0]) for group in groups.groups]
    return [to_hit(point) for point in response.points]


def dense_query(query: str, inference: Inference) -> list[float] | models.Document:
    if inference == "cloud":
        return models.Document(text=query, model=DENSE_MODEL)
    return embed_query(query)


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
        url=str(payload.get("url", "")),
        title=str(payload.get("title", "")),
        section=str(payload.get("section", "")),
        content_type=str(payload.get("content_type", "")),
    )
