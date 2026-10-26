import csv
import gzip
import io
import json
import random
import tarfile
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_sample_datasets.core.download import download_file

BASE_URL = "https://amazon-berkeley-objects.s3.amazonaws.com/"
LISTINGS_FILE = "abo-listings.tar"
IMAGES_FILE = "images.csv.gz"
SOURCES = {
    LISTINGS_FILE: "archives/abo-listings.tar",
    IMAGES_FILE: "images/metadata/images.csv.gz",
}
IMAGE_URL = BASE_URL + "images/small/"
PHONE_CASE_TYPE = "CELLULAR_PHONE_CASE"
# Main images that are placeholders rather than product photos, found by comparing perceptual hashes
# of every main image: "Image Coming Soon", "Product Image Coming Soon", the AmazonBasics logo,
# a "Satisfaction Guarantee" card, and an "FPO" graphic. Together they cover 905 products.
PLACEHOLDER_IMAGE_IDS = frozenset(
    {
        "31OikUsN7qL",
        "31odnBmkMML",
        "51S1fx8uSCL",
        "51UFZGN-IiL",
        "61kVdTXTJ4L",
        "31Fve1fdFWL",
        "31VRqnfhp6L",
        "31lJBp1dALL",
        "31ydHQXAUjL",
        "51l2v7JF-LL",
        "31OuRX-HAIL",
        "51zboj55JqL",
        "61OnkGVhMWL",
        "61qNcMpEZGL",
        "61w4fs2ESyL",
        "61xDi-Uv1kL",
        "41KSWZxricL",
        "51M39MBAwDL",
        "51ZZTwJMfpL",
        "21Qlz9BiTzL",
        "21cjGwLXbzL",
    }
)
# English variants in order of preference when a listing has several.
ENGLISH_TAGS = ("en_US", "en_GB", "en_IN", "en_CA", "en_AU")


@dataclass(frozen=True)
class Product:
    id: str
    item_id: str
    name: str
    brand: str | None
    color: str | None
    product_type: str | None
    country: str | None
    image_url: str

    @property
    def payload(self) -> dict[str, str]:
        fields = {
            "item_id": self.item_id,
            "name": self.name,
            "brand": self.brand,
            "color": self.color,
            "product_type": self.product_type,
            "country": self.country,
            "image_url": self.image_url,
        }
        return {key: value for key, value in fields.items() if value}


def download(raw_dir: Path) -> None:
    for name, path in SOURCES.items():
        download_file(BASE_URL + path, raw_dir / name)


def load(raw_dir: Path, phone_case_cap: int, seed: int) -> list[Product]:
    images = image_paths(raw_dir / IMAGES_FILE)
    products: dict[str, Product] = {}
    for listing in listings(raw_dir / LISTINGS_FILE):
        product = to_product(listing, images)
        if product and product.item_id not in products:
            products[product.item_id] = product
    return cap_phone_cases(one_per_image(products.values()), phone_case_cap, seed)


def listings(archive: Path) -> Iterator[dict[str, Any]]:
    """Read every listing from the gzipped JSON files inside the archive, without extracting it."""
    with tarfile.open(archive) as tar:
        members = sorted(
            (m for m in tar.getmembers() if m.isfile() and m.name.endswith(".json.gz")),
            key=lambda m: m.name,
        )
        for member in members:
            handle = tar.extractfile(member)
            if handle is None:
                continue
            with gzip.open(io.BytesIO(handle.read()), "rt", encoding="utf-8") as lines:
                for line in lines:
                    yield json.loads(line)


def image_paths(path: Path) -> dict[str, str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {row["image_id"]: row["path"] for row in csv.DictReader(handle)}


def to_product(listing: dict[str, Any], images: dict[str, str]) -> Product | None:
    name = english(listing.get("item_name"))
    image_id = listing.get("main_image_id") or ""
    image_path = images.get(image_id)
    if not name or not image_path or image_id in PLACEHOLDER_IMAGE_IDS:
        return None
    product_types = listing.get("product_type") or []
    return Product(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"abo:{listing['item_id']}")),
        item_id=listing["item_id"],
        name=name,
        brand=english(listing.get("brand")),
        color=english(listing.get("color")),
        product_type=product_types[0].get("value") if product_types else None,
        country=listing.get("country"),
        image_url=IMAGE_URL + image_path,
    )


def english(values: list[dict[str, Any]] | None) -> str | None:
    """Return the English value of a language-tagged field, preferring US English."""
    by_tag = {value.get("language_tag"): value.get("value") for value in values or []}
    for tag in ENGLISH_TAGS:
        if by_tag.get(tag):
            return str(by_tag[tag]).strip()
    return next(
        (str(v).strip() for t, v in by_tag.items() if t and t.startswith("en") and v),
        None,
    )


def one_per_image(products: Iterable[Product]) -> list[Product]:
    """Keep one product per main image, the one with the shortest name.

    Products that share a main image are usually size or color variants, and the shortest
    name is usually the parent title without the size.
    """
    kept: dict[str, Product] = {}
    for product in products:
        current = kept.get(product.image_url)
        if current is None or (len(product.name), product.item_id) < (
            len(current.name),
            current.item_id,
        ):
            kept[product.image_url] = product
    return sorted(kept.values(), key=lambda p: p.item_id)


def cap_phone_cases(products: list[Product], cap: int, seed: int) -> list[Product]:
    cases = [p for p in products if p.product_type == PHONE_CASE_TYPE]
    kept = {p.item_id for p in random.Random(seed).sample(cases, min(cap, len(cases)))}
    return [
        p for p in products if p.product_type != PHONE_CASE_TYPE or p.item_id in kept
    ]
