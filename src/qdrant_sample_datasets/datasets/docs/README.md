# Qdrant Docs

A sample hybrid search dataset built from the content of [qdrant.tech](https://qdrant.tech): the documentation, the technical articles, and the Qdrant Academy course. It is meant to be published as a snapshot on the [datasets page](https://qdrant.tech/documentation/datasets/) so people can try hybrid search on a real corpus without running any inference themselves.

## What Is in the Collection

Each point is one chunk of a page. The chunker follows the page's heading tree. A section that fits the token budget becomes one chunk, with its subsection headings kept in the text. A section that is too large is split into its intro and its subsections, and a section with no subsections is split at paragraph boundaries. Chunks under 20 words are merged into a neighboring chunk from the same page.

| Item | Value |
|---|---|
| Points | 3593 chunks from 504 pages |
| Dense vector `dense` | `intfloat/multilingual-e5-small`, 384 dimensions, cosine |
| Lexical retriever | BM25 scoring on the full-text index of the `text` payload field (Qdrant 1.20 or later) |
| Payload indexes | `url` keyword, `content_type` keyword, `has_code` bool, `text` full-text with BM25 scoring, `code` tokenizer, English stemming and stopwords |

The `text` field starts with a heading line, the page title and the headings above the chunk joined with " > ", followed by the chunk body in Markdown. The dense vector embeds `text` with code blocks, image references, and link URLs removed. Code blocks stay in `text`, where BM25 and the `code` tokenizer make identifiers such as `ef_construct` or `efConstruct` searchable. Code examples keep only the Python version when one exists.

Payload fields:

| Field | Type | Index | Example |
|---|---|---|---|
| `url` | string | Keyword | `/documentation/search/hybrid-queries/` |
| `title` | string | None | `Hybrid Queries` |
| `section` | string | None | `Hybrid Search > Choosing a Fusion Method` |
| `content_type` | string | Keyword | `documentation`, `article`, or `course` |
| `has_code` | bool | Bool | `true` when the chunk contains at least one code block |
| `text` | string | Full-text with BM25 | Heading line, then the chunk in Markdown, including code blocks |

The full-text index on `text` uses this configuration:

| Setting | Value |
|---|---|
| Tokenizer | `code` |
| Lowercase | Yes |
| Stemmer | English Snowball |
| Stopwords | English |
| Scoring | BM25, with `k` and `b` set per query (defaults 1.2 and 0.75) |

Point IDs are UUIDv5 values derived from the page URL and the position of the chunk in the page, so regenerating the dataset keeps the IDs of unchanged pages stable.

## Requirements

- Qdrant 1.20 or later. The BM25 text query and `scoring` on the full-text index were introduced in 1.20.
- A Python client with the `TextQuery` model, which is any `qdrant-client` release from 1.20 on.

## Querying the Dataset

On Qdrant Cloud, the dense model is free with Cloud Inference, and the BM25 side needs no inference at all. A hybrid query is one request:

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(
    url="https://YOUR-CLUSTER.cloud.qdrant.io", api_key="YOUR-KEY", cloud_inference=True
)

question = "how do I tune ef_construct for better recall"

results = client.query_points(
    collection_name="qdrant-docs",
    prefetch=[
        models.Prefetch(
            query=models.Document(
                text=question, model="intfloat/multilingual-e5-small"
            ),
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

The example fuses both retrievers with equal weights. Tune the RRF weights for your queries.

Cloud Inference adds the `query:` and `passage:` prefixes that E5 models expect, so no prefix handling is needed on the client.

## Building the Dataset

Run these from the repository root. The source is the content folder of the [landing page repository](https://github.com/qdrant/landing_page):

```bash
git clone --depth 1 https://github.com/qdrant/landing_page.git /tmp/landing_page
uv run qdrant-datasets docs prepare --content /tmp/landing_page/qdrant-landing/content
uv run qdrant-datasets docs build --inference cloud
uv run qdrant-datasets docs query "what is the difference between rrf and dbsf" --mode hybrid --inference cloud
uv run qdrant-datasets docs snapshot --inference cloud
```

`prepare` writes the chunks to `data/docs/records.jsonl` and prints the number of pages and chunks and the token size distribution. Each record also carries the `dense_text` and `tokens` fields used during the build. Those are not stored in the payload. `snapshot` saves the file to `snapshots/docs/`.

`query` supports four modes: `dense`, `text`, `hybrid`, and `groups`, which returns one result per page.

`--inference cloud` sends `Document` objects so Qdrant Cloud Inference computes the dense vectors. Use it against a Qdrant Cloud cluster for the published snapshot, so the stored vectors come from the same model deployment people query with. `--inference local` computes the vectors on your machine and needs the `local` extra, for example `uv run --extra local qdrant-datasets docs build --inference local`.

### What the Chunker Does with the Source

- Skips `headless` and `snippets` directories, draft pages, `delimiter` and `external-link` navigation entries, pages excluded from the sitemap, and pages with fewer than 40 words of body.
- Resolves `{{< code-snippet >}}` shortcodes to the Python snippet, falling back to the HTTP, bash, or shell variant. Every other shortcode is dropped.
- Keeps only the Python version of SDK examples. Pages without a Python example keep their code as it is.
- Treats the first H1 as the page title. On pages that also use H1 for sections, every heading moves down one level.
- Never splits a code block, and does not count code toward the token budget.
- Strips HTML tags from prose but keeps their text, so `<aside>` notes survive. Inline `<svg>`, `<script>`, `<iframe>`, and `<video>` blocks are removed.
- Keeps Markdown tables and lists. A table that exceeds the token budget is split by rows, and the header row is repeated in each part.
- Uses a budget of 480 E5 tokens per chunk, and allows up to 505 tokens when merging a chunk under 20 words.
- Builds the page URL the same way Hugo does: path based, with `url:` and `slug:` frontmatter overrides.
