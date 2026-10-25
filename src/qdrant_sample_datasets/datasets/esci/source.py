import html
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

SOURCE_URL = "https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset/"
EXAMPLES_FILE = "shopping_queries_dataset_examples.parquet"
PRODUCTS_FILE = "shopping_queries_dataset_products.parquet"
TAG_RE = re.compile(r"<[^>]+>")

# The US queries of the small version, test split, and every product they judge.
JUDGMENTS_SQL = """
    select query_id, query, product_id, esci_label
    from read_parquet(?)
    where product_locale = 'us' and small_version = 1 and split = 'test'
"""
PRODUCTS_SQL = """
    select product_id, product_title, product_brand, product_color, product_bullet_point
    from read_parquet(?)
    where product_locale = 'us' and product_id in (select product_id from judgments)
    order by product_id
"""


@dataclass(frozen=True)
class Product:
    id: str
    product_id: str
    title: str
    brand: str | None
    color: str | None
    text: str

    @property
    def payload(self) -> dict[str, str]:
        fields = {
            "product_id": self.product_id,
            "title": self.title,
            "brand": self.brand,
            "color": self.color,
        }
        return {key: value for key, value in fields.items() if value} | {
            "text": self.text
        }


@dataclass(frozen=True)
class Query:
    query_id: int
    query: str
    judgments: dict[str, str]


def download(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in (EXAMPLES_FILE, PRODUCTS_FILE):
        target = raw_dir / name
        if target.exists():
            continue
        partial = target.with_suffix(".part")
        with httpx.stream(
            "GET", SOURCE_URL + name, follow_redirects=True, timeout=None
        ) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for data in response.iter_bytes():
                    handle.write(data)
        partial.rename(target)


def load(raw_dir: Path) -> tuple[list[Product], list[Query]]:
    # Imported here because duckdb is the optional `esci` extra.
    import duckdb

    con = duckdb.connect()
    judgments = con.execute(JUDGMENTS_SQL, [str(raw_dir / EXAMPLES_FILE)]).fetchall()
    con.execute(
        "create temp table judgments as select * from (values (null::varchar)) t(product_id) where false"
    )
    con.executemany(
        "insert into judgments values (?)", [(row[2],) for row in judgments]
    )
    rows = con.execute(PRODUCTS_SQL, [str(raw_dir / PRODUCTS_FILE)]).fetchall()
    return [to_product(*row) for row in rows], to_queries(judgments)


def to_product(
    product_id: str,
    title: str | None,
    brand: str | None,
    color: str | None,
    bullets: str | None,
) -> Product:
    title = clean(title)
    brand = clean(brand) or None
    lines = [title, *([f"Brand: {brand}"] if brand else []), clean(bullets)]
    return Product(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"esci:{product_id}")),
        product_id=product_id,
        title=title,
        brand=brand,
        color=clean(color) or None,
        text="\n".join(line for line in lines if line),
    )


def to_queries(judgments: list[tuple[Any, ...]]) -> list[Query]:
    queries: dict[int, Query] = {}
    for query_id, query, product_id, label in judgments:
        queries.setdefault(query_id, Query(query_id, query, {})).judgments[
            product_id
        ] = label
    return sorted(queries.values(), key=lambda query: query.query_id)


def clean(text: str | None) -> str:
    """Strip HTML, then collapse whitespace on each line and drop empty lines."""
    if not text:
        return ""
    text = html.unescape(TAG_RE.sub(" ", text))
    lines = (re.sub(r"\s+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)
