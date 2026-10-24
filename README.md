# Qdrant Docs Dataset

A sample hybrid search dataset built from the content of [qdrant.tech](https://qdrant.tech): the documentation, the technical articles, and the Qdrant Academy course. It is meant to be published as a snapshot on the [datasets page](https://qdrant.tech/documentation/datasets/) so people can try hybrid search on a real corpus without running any inference themselves.

## What is in the collection

Each point is one chunk of a page. Pages are split at H2 headings, then at H3 headings and paragraph boundaries until every chunk fits the dense model's context window.

| Item | Value |
|---|---|
| Points | 3417 chunks from 504 pages |
| Dense vector `dense` | `intfloat/multilingual-e5-small`, 384 dimensions, cosine |
| Lexical retriever | BM25 scoring on the full-text index of the `text` payload field (Qdrant 1.20 or later) |
| Payload indexes | `url` keyword, `content_type` keyword, `text` full-text with BM25 scoring, `code` tokenizer, English stemming and stopwords |

The dense vector embeds the page title, the heading path, and the prose of the chunk. Code blocks are left out of the dense input and kept in the `text` payload, where the `code` tokenizer makes identifiers such as `ef_construct` or `efConstruct` searchable.

Payload fields:

| Field | Type | Example |
|---|---|---|
| `url` | keyword | `/documentation/search/hybrid-queries/` |
| `title` | string | `Hybrid Queries` |
| `section` | string | `Hybrid Search > Choosing a Fusion Method` |
| `content_type` | keyword | `documentation`, `article`, or `course` |
| `chunk_index` | integer | `3` |
| `text` | text | The chunk in Markdown, including code blocks |

Point IDs are UUIDv5 values derived from the page URL and the chunk index, so regenerating the dataset keeps IDs stable.

## Requirements

- Qdrant 1.20 or later. The BM25 text query and `scoring` on the full-text index were introduced in 1.20.
- A Python client with the `TextQuery` model, which is any `qdrant-client` release from 1.20 on.

## Querying the dataset

On Qdrant Cloud, the dense model is free with Cloud Inference, and the BM25 side needs no inference at all. A hybrid query is one request:

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="https://YOUR-CLUSTER.cloud.qdrant.io", api_key="YOUR-KEY")

question = "how do I tune ef_construct for better recall"

results = client.query_points(
    collection_name="qdrant-docs",
    prefetch=[
        models.Prefetch(
            query=models.Document(text=question, model="intfloat/multilingual-e5-small"),
            using="dense",
            limit=20,
        ),
        models.Prefetch(query=models.TextQuery(text=question), using="text", limit=20),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=5,
)
```

To get one result per page instead of per chunk, use `query_points_groups` with `group_by="url"`. The same prefetch and fusion apply.

Cloud Inference adds the `query:` and `passage:` prefixes that E5 models expect, so no prefix handling is needed on the client.

## Building the dataset

The repo is a `uv` project. The `qdrant-docs-dataset` command has four steps:

```bash
git clone --depth 1 https://github.com/qdrant/landing_page.git
uv run qdrant-docs-dataset chunk --content landing_page/qdrant-landing/content --out chunks.jsonl
uv run qdrant-docs-dataset build --chunks chunks.jsonl --url http://localhost:6333 --inference local
uv run qdrant-docs-dataset query "what is the difference between rrf and dbsf" --mode hybrid
uv run qdrant-docs-dataset snapshot --out snapshots
```

`--inference local` computes the dense vectors on your machine with `sentence-transformers`. `--inference cloud` sends `Document` objects so Qdrant Cloud Inference computes them. Use `cloud` against a Qdrant Cloud cluster for the published snapshot so the stored vectors come from the same model deployment people query with. `QDRANT_URL` and `QDRANT_API_KEY` are read from the environment when the flags are omitted.

`chunk` prints the number of pages and chunks and the token size distribution. Chunks written to `chunks.jsonl` also carry the `dense_text` and `tokens` fields used during the build. Those are not stored in the payload.

### What the chunker does with the source

- Skips `headless` and `snippets` directories, draft pages, `delimiter` and `external-link` navigation entries, pages excluded from the sitemap, and pages with fewer than 40 words of body.
- Resolves `{{< code-snippet >}}` shortcodes to the Python snippet, falling back to the HTTP or bash variant. Every other shortcode is dropped.
- Strips HTML tags from prose but keeps their text, so `<aside>` notes survive. Inline `<svg>`, `<script>`, `<iframe>`, and `<video>` blocks are removed.
- Keeps Markdown tables and lists. A table that exceeds the token budget is split by rows, and the header row is repeated in each part.
- Builds the page URL the same way Hugo does: path based, with `url:` and `slug:` frontmatter overrides.
