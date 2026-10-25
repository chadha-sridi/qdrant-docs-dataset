# Amazon ESCI Products

A product search collection built from the [Shopping Queries Dataset](https://github.com/amazon-science/esci-data) (ESCI) by Amazon Science. It ships with real shopping queries and graded relevance judgments, so you can measure how well dense, keyword, and hybrid search rank products, not just try them.

## What Is in the Collection

The collection holds every product judged for the US test queries of the dataset's small version. Each point is one product.

| Item | Value |
|---|---|
| Points | 164,900 products |
| Dense vector `dense` | `intfloat/multilingual-e5-small`, 384 dimensions, cosine |
| Lexical retriever | BM25 scoring on the full-text index of the `text` payload field (Qdrant 1.20 or later) |
| Relevance judgments | 181,701 judgments for 8,956 queries, in `queries.jsonl` |

The `text` field holds the product title, a `Brand:` line when the brand is known, and the bullet points. The dense vector embeds `text`. E5 small reads up to 512 tokens, so for about 3% of products the vector covers only the start of the text, while BM25 indexes all of it. Product descriptions are not included.

Payload fields:

| Field | Type | Index | Example |
|---|---|---|---|
| `product_id` | string | Keyword | `B08LPC9JTB` |
| `title` | string | None | `2 Pieces 6 AN ORB Male to 5/16 SAE Quick-Disconnect Female Fuel Pump Adapter Push-On EFI Fitting Aluminum O-ring` |
| `brand` | string | Keyword | `BBTO`, missing on about 5% of products |
| `color` | string | Keyword | `Black`, missing on about 31% of products |
| `text` | string | Full-text with BM25 | Title, brand line, then bullet points |

Missing brands and colors are left out of the payload rather than stored as empty values.

The full-text index on `text` uses this configuration:

| Setting | Value |
|---|---|
| Tokenizer | `word` |
| Lowercase | Yes |
| Stemmer | English Snowball |
| Stopwords | English |
| Scoring | BM25, with `k` and `b` set per query (defaults 1.2 and 0.75) |

Point IDs are UUIDv5 values derived from the product ID.

## Relevance Judgments

`queries.jsonl` is meant to be published next to the snapshot. Each line is one query with its judged products:

```json
{"query_id": 2, "query": "!qscreen fence without holes", "judgments": {"B07DHX8YH2": "E", "B07DS1YCRZ": "S", "B07D7TBSGH": "I"}}
```

Each query and product pair was labeled by hand with one of four labels. The gains are the ones the ESCI paper uses for normalized discounted cumulative gain (nDCG):

| Label | Meaning | Gain |
|---|---|---|
| `E`, Exact | The product matches the query | 1.0 |
| `S`, Substitute | The product doesn't fully match, but could replace an exact match | 0.1 |
| `C`, Complement | The product doesn't match, but goes with an exact match | 0.01 |
| `I`, Irrelevant | The product doesn't help | 0.0 |

Each query has about 20 judged products, and all of them are in the collection. Products a search returns without a judgment count as irrelevant. The queries are real Amazon searches, so many are short, misspelled, or niche.

## Requirements

- Qdrant 1.20 or later. The BM25 text query and `scoring` on the full-text index were introduced in 1.20.
- A Python client with the `TextQuery` model, which is any `qdrant-client` release from 1.20 on.

## Querying the Dataset

On Qdrant Cloud, the dense model is free with Cloud Inference, and the BM25 side needs no inference. This hybrid query keeps only one brand, applying the filter in both prefetches:

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(
    url="https://YOUR-CLUSTER.cloud.qdrant.io", api_key="YOUR-KEY", cloud_inference=True
)

question = "usb c headphones"
samsung_only = models.Filter(
    must=[models.FieldCondition(key="brand", match=models.MatchValue(value="Samsung"))]
)

results = client.query_points(
    collection_name="esci-products",
    prefetch=[
        models.Prefetch(
            query=models.Document(
                text=question, model="intfloat/multilingual-e5-small"
            ),
            using="dense",
            filter=samsung_only,
            limit=50,
        ),
        models.Prefetch(
            query=models.TextQuery(text=question),
            using="text",
            filter=samsung_only,
            limit=50,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=10,
)
```

The example fuses both retrievers with equal weights. Tune the RRF weights for your queries.

## Building the Dataset

Run these from the repository root. Reading the source Parquet files needs the `esci` extra:

```bash
uv run --extra esci qdrant-datasets esci prepare
uv run qdrant-datasets esci build --inference cloud
uv run qdrant-datasets esci query "usb c headphones" --mode hybrid --inference cloud
uv run qdrant-datasets esci evaluate --inference cloud
uv run qdrant-datasets esci snapshot --inference cloud
```

`prepare` downloads the two source files, about 1.2 GB, to `data/esci/raw/` the first time. It writes the products to `data/esci/records.jsonl` and the judgments to `data/esci/queries.jsonl`, then prints the counts and the token size distribution.

`evaluate` runs every query in `queries.jsonl` and prints nDCG@10 for dense, text, and hybrid search. `--mode` picks one search mode, and `--sample` evaluates a random subset of queries.

`snapshot` saves the file to `snapshots/esci/`. Publish `data/esci/queries.jsonl` next to it.

## License and Attribution

The Shopping Queries Dataset is released by Amazon Science under the [Apache 2.0 license](https://github.com/amazon-science/esci-data/blob/main/LICENSE). If you use it in research, cite the paper:

> Chandan K. Reddy, Lluís Màrquez, Fran Valero, Nikhil Rao, Hugo Zaragoza, Sambaran Bandyopadhyay, Arnab Biswas, Anlu Xing, and Karthik Subbian. 2022. [Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search](https://arxiv.org/abs/2206.06588). arXiv:2206.06588.
