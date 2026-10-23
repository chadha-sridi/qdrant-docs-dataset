from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import httpx
from qdrant_client import QdrantClient, models

from qdrant_docs_dataset.chunk import Chunk
from qdrant_docs_dataset.embed import DENSE_MODEL, DENSE_SIZE, embed_passages

Inference = Literal["local", "cloud"]

DENSE_VECTOR = "dense"
TEXT_FIELD = "text"
BATCH_SIZE = 64
# The collection is small, so build the HNSW graph instead of staying on the plain index.
INDEXING_THRESHOLD_KB = 1000


def build_collection(
    client: QdrantClient, collection: str, chunks: list[Chunk], inference: Inference
) -> None:
    create_collection(client, collection)
    for batch in batched(chunks, BATCH_SIZE):
        client.upsert(
            collection_name=collection, points=to_points(batch, inference), wait=True
        )
    create_indexes(client, collection)


def create_collection(client: QdrantClient, collection: str) -> None:
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(
                size=DENSE_SIZE, distance=models.Distance.COSINE
            )
        },
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=INDEXING_THRESHOLD_KB
        ),
    )


def create_indexes(client: QdrantClient, collection: str) -> None:
    for field in ("url", "content_type"):
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD
            ),
            wait=True,
        )
    client.create_payload_index(
        collection_name=collection,
        field_name=TEXT_FIELD,
        field_schema=models.TextIndexParams(
            type=models.TextIndexType.TEXT,
            tokenizer=models.TokenizerType.CODE,
            lowercase=True,
            stemmer=models.SnowballParams(
                type=models.Snowball.SNOWBALL, language=models.SnowballLanguage.ENGLISH
            ),
            stopwords=models.Language.ENGLISH,
            scoring=models.TextScoringParams(type=models.TextScoringType.BM25),
        ),
        wait=True,
    )


def to_points(chunks: list[Chunk], inference: Inference) -> list[models.PointStruct]:
    vectors = dense_vectors(chunks, inference)
    return [
        models.PointStruct(
            id=chunk.id, vector={DENSE_VECTOR: vector}, payload=chunk.payload
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]


def dense_vectors(
    chunks: list[Chunk], inference: Inference
) -> list[list[float]] | list[models.Document]:
    if inference == "cloud":
        return [
            models.Document(text=chunk.dense_text, model=DENSE_MODEL)
            for chunk in chunks
        ]
    return embed_passages([chunk.dense_text for chunk in chunks])


def batched(items: list[Chunk], size: int) -> Iterator[list[Chunk]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def download_snapshot(
    client: QdrantClient, url: str, api_key: str | None, collection: str, out_dir: Path
) -> Path:
    description = client.create_snapshot(collection_name=collection, wait=True)
    if description is None:
        raise RuntimeError("snapshot creation returned no description")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / description.name
    headers = {"api-key": api_key} if api_key else {}
    with (
        httpx.stream(
            "GET",
            f"{url}/collections/{collection}/snapshots/{description.name}",
            headers=headers,
            timeout=None,
        ) as response,
        target.open("wb") as handle,
    ):
        response.raise_for_status()
        for data in response.iter_bytes():
            handle.write(data)
    return target
