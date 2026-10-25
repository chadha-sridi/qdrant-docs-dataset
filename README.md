# Qdrant Sample Datasets

This repository builds the sample datasets that Qdrant publishes as snapshots on the [datasets page](https://qdrant.tech/documentation/datasets/). Each dataset is prepared from a public source, embedded, loaded into a Qdrant collection, and exported as a snapshot that anyone can restore and query.

## Datasets

| Dataset | Description | Points | Vectors | Dataset card |
|---|---|---|---|---|
| `docs` | Qdrant documentation, articles, and course, chunked by section | 3593 | Dense E5 small, plus BM25 on a full-text index | [Qdrant Docs](src/qdrant_sample_datasets/datasets/docs/README.md) |
| `esci` | Amazon products with real shopping queries and graded relevance judgments | 164,900 | Dense E5 small, plus BM25 on a full-text index | [Amazon ESCI Products](src/qdrant_sample_datasets/datasets/esci/README.md) |

Each dataset card describes the collection, the payload fields and indexes, the source license, a query example, and how to build it.

## Setup

You need Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Computing embeddings on your machine with `--inference local` needs the `local` extra:

```bash
uv sync --extra local
```

Some datasets need an extra of their own to prepare their source data, such as `esci`. Their dataset cards say so, and list the Qdrant version their collection needs.

## Usage

Every dataset goes through the same four steps:

```bash
uv run qdrant-datasets list
uv run qdrant-datasets <dataset> prepare [dataset options]
uv run qdrant-datasets <dataset> build --inference cloud
uv run qdrant-datasets <dataset> query "your question" --inference cloud
uv run qdrant-datasets <dataset> snapshot --inference cloud
```

| Step | What it does | Output |
|---|---|---|
| `prepare` | Reads the source data and writes cleaned records | `data/<dataset>/records.jsonl` |
| `build` | Recreates the collection, upserts the records, and creates the payload indexes | A Qdrant collection |
| `query` | Runs a search to check the collection | Results in the terminal |
| `snapshot` | Creates a collection snapshot and downloads it | `snapshots/<dataset>/` |
| `evaluate` | Scores search quality against the relevance judgments, for datasets that ship them | nDCG in the terminal |

`build`, `query`, and `snapshot` connect to the cluster in `QDRANT_URL` with the key in `QDRANT_API_KEY`, or to `http://localhost:6333` when those aren't set. `--collection` overrides the default collection name. `--inference cloud` lets Qdrant Cloud Inference compute the embeddings, and `--inference local` computes them on your machine. Run any step with `--help` to see its options.

## Repository Layout

```text
src/qdrant_sample_datasets/
├── cli.py                 # qdrant-datasets command
├── core/
│   ├── dataset.py         # Dataset base class, CollectionSpec, Hit
│   ├── qdrant.py          # Connect, build a collection, download a snapshot
│   ├── records.py         # JSONL helpers and batching
│   ├── text_models.py     # Text embedding models, tokens, local or cloud vectors
│   └── local_embeddings.py
└── datasets/
    ├── __init__.py        # Registry of available datasets
    ├── docs/              # One folder per dataset, with its README.md card
    └── esci/
```

`data/` and `snapshots/` hold generated files and are ignored by git.

## Adding a Dataset

1. Create a folder under `src/qdrant_sample_datasets/datasets/` with a subclass of `Dataset` from `core/dataset.py`. It sets the dataset name, a description, and a `CollectionSpec` with the vectors and payload indexes. It implements `prepare`, `points`, and `search`, and can add its own options to the `prepare` and `query` steps. A dataset with relevance judgments sets `supports_evaluation` and implements `evaluate`.
2. Register an instance in `src/qdrant_sample_datasets/datasets/__init__.py`.
3. Write the dataset card as `README.md` in the dataset folder, and add a row to the table in this file.
4. Add dependencies that only this dataset needs as an optional extra in `pyproject.toml`, and import them inside the functions that use them, so the other datasets don't need them.
5. Confirm that the source license allows redistribution, and credit the source in the dataset card as its license requires.
