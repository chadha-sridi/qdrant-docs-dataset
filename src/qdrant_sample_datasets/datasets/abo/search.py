from typing import Literal

from qdrant_client import QdrantClient, models

from qdrant_sample_datasets.datasets.abo.schema import (
    IMAGE_MODEL,
    IMAGE_VECTOR,
    NAME_FIELD,
    TEXT_MODEL,
)

Mode = Literal["image", "text", "hybrid"]
MODES: tuple[Mode, ...] = ("image", "text", "hybrid")
PREFETCH_LIMIT = 50


def search_by_text(
    client: QdrantClient, collection: str, query: str, mode: Mode, limit: int
) -> list[models.ScoredPoint]:
    """Search with a text query: CLIP text against the image vector, BM25 on names, or both fused."""
    clip = models.Document(text=query, model=TEXT_MODEL)
    bm25 = models.TextQuery(text=query)
    if mode == "image":
        return client.query_points(
            collection, query=clip, using=IMAGE_VECTOR, limit=limit
        ).points
    if mode == "text":
        return client.query_points(
            collection, query=bm25, using=NAME_FIELD, limit=limit
        ).points
    return client.query_points(
        collection,
        prefetch=[
            models.Prefetch(query=clip, using=IMAGE_VECTOR, limit=PREFETCH_LIMIT),
            models.Prefetch(query=bm25, using=NAME_FIELD, limit=PREFETCH_LIMIT),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
    ).points


def search_by_image(
    client: QdrantClient, collection: str, image_url: str, limit: int
) -> list[models.ScoredPoint]:
    """Find products whose main image looks like the given image."""
    image = models.Image(image=image_url, model=IMAGE_MODEL)
    return client.query_points(
        collection, query=image, using=IMAGE_VECTOR, limit=limit
    ).points
