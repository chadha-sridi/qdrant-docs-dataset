from collections.abc import Iterable, Iterator
from pathlib import Path

import httpx
from qdrant_client import QdrantClient, models

from qdrant_sample_datasets.core.dataset import CollectionSpec, Inference


def connect(
    url: str, api_key: str | None, inference: Inference = "local"
) -> QdrantClient:
    return QdrantClient(
        url=url, api_key=api_key, timeout=120, cloud_inference=inference == "cloud"
    )


def build_collection(
    client: QdrantClient,
    name: str,
    spec: CollectionSpec,
    batches: Iterable[list[models.PointStruct]],
) -> int:
    """Recreate the collection, upsert every batch, create payload indexes, and return the point count."""
    recreate_collection(client, name, spec)
    for batch in batches:
        client.upsert(collection_name=name, points=batch, wait=True)
    for field_name, schema in spec.payload_indexes.items():
        client.create_payload_index(
            collection_name=name, field_name=field_name, field_schema=schema, wait=True
        )
    return client.count(name).count


def limit_points(
    batches: Iterable[list[models.PointStruct]], limit: int | None
) -> Iterator[list[models.PointStruct]]:
    """Stop the stream of point batches after `limit` points, or pass it through when there is no limit."""
    remaining = limit
    for batch in batches:
        if remaining is not None:
            batch = batch[:remaining]
            remaining -= len(batch)
        if batch:
            yield batch
        if remaining == 0:
            return


def recreate_collection(client: QdrantClient, name: str, spec: CollectionSpec) -> None:
    if client.collection_exists(name):
        client.delete_collection(name)
    optimizers = (
        models.OptimizersConfigDiff(indexing_threshold=spec.indexing_threshold_kb)
        if spec.indexing_threshold_kb is not None
        else None
    )
    client.create_collection(
        collection_name=name, vectors_config=spec.vectors, optimizers_config=optimizers
    )


def download_snapshot(
    client: QdrantClient, url: str, api_key: str | None, collection: str, out_dir: Path
) -> Path:
    description = client.create_snapshot(collection_name=collection, wait=True)
    if description is None:
        raise RuntimeError("snapshot creation returned no description")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / description.name
    headers = {"api-key": api_key} if api_key else {}
    snapshot_url = f"{url}/collections/{collection}/snapshots/{description.name}"
    with (
        httpx.stream("GET", snapshot_url, headers=headers, timeout=None) as response,
        target.open("wb") as handle,
    ):
        response.raise_for_status()
        for data in response.iter_bytes():
            handle.write(data)
    return target
