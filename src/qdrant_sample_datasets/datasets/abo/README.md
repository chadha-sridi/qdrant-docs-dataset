# Amazon Berkeley Objects Products

A multimodal product collection built from [Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) (ABO). Each product has a CLIP embedding of its main image, so you can find products by describing them in words, by showing a similar image, or by combining CLIP with BM25 keyword search on product names.

## What Is in the Collection

| Item | Value |
|---|---|
| Points | 44,933 products across 524 product types |
| Vector, unnamed | `qdrant/clip-vit-b-32-vision` on the product's main image, 512 dimensions, cosine |
| Lexical retriever | BM25 scoring on the full-text index of the `name` payload field (Qdrant 1.20 or later) |

ABO products have about five images each. The collection embeds only the main image, the primary product photo that each listing designates, and doesn't use the others.

The collection keeps every ABO product that has an English name and a main image. ABO lists the same product in several marketplaces, and the collection keeps one listing per product. Products that share a main image, usually size or color variants of one item, are kept once, under the shortest of their names, which is usually the title without the size. Phone cases make up half of the source catalog, almost all from the Indian marketplace, so the collection keeps 1,000 of them, chosen at random. Products whose main image is a placeholder rather than a photo are left out: "Image Coming Soon" graphics, the AmazonBasics logo, a "Satisfaction Guarantee" card, and an "FPO" graphic.

Payload fields:

| Field | Type | Index | Example |
|---|---|---|---|
| `item_id` | string | Keyword | `B07CTPR73M` |
| `name` | string | Full-text with BM25 | `Ravenna Home Jacobsen Contemporary Nailhead Trim Accent Chair` |
| `brand` | string | Keyword | `Ravenna Home`, missing on about 7% of products |
| `color` | string | Keyword | `Stone Brown`, missing on about 43% of products |
| `product_type` | string | Keyword | `CHAIR` |
| `country` | string | Keyword | `US`, the marketplace of the listing |
| `image_url` | string | None | `https://amazon-berkeley-objects.s3.amazonaws.com/images/small/a4/a47ddc30.jpg` |

Names, brands, and colors use the English value of each listing, preferring US English when a listing has several. Missing values are left out of the payload. `image_url` points to the downscaled image on ABO's public S3 bucket, at most 256 pixels on the longest side. The original-resolution image is at the same path under `images/original/`. The snapshot stores the URLs, not the images.

The full-text index on `name` uses the `word` tokenizer, lowercasing, the English Snowball stemmer, English stopwords, and BM25 scoring with `k` and `b` set per query.

Point IDs are UUIDv5 values derived from the ABO item ID.

## Requirements

- Qdrant 1.20 or later. The BM25 text query and `scoring` on the full-text index were introduced in 1.20.
- Qdrant Cloud Inference with the CLIP models `qdrant/clip-vit-b-32-vision` and `qdrant/clip-vit-b-32-text` for queries that use the image vector.

## Querying the Dataset

The collection has a single unnamed vector, so queries against it need no `using`. The CLIP text model and the CLIP image model share one embedding space, so a text query embedded with the text model searches the image vector directly. This hybrid query fuses that with BM25 on the product names:

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(
    url="https://YOUR-CLUSTER.cloud.qdrant.io", api_key="YOUR-KEY", cloud_inference=True
)

question = "brown leather armchair"

results = client.query_points(
    collection_name="amazon-berkeley-objects",
    prefetch=[
        models.Prefetch(
            query=models.Document(text=question, model="qdrant/clip-vit-b-32-text"),
            limit=50,
        ),
        models.Prefetch(query=models.TextQuery(text=question), using="name", limit=50),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=10,
)
```

To find products that look like a picture, embed the picture with the image model and search the same vector:

```python
results = client.query_points(
    collection_name="amazon-berkeley-objects",
    query=models.Image(
        image="https://example.com/photo.jpg", model="qdrant/clip-vit-b-32-vision"
    ),
    limit=10,
)
```

The hybrid example fuses both retrievers with equal weights. Tune the RRF weights for your queries.

## Building the Dataset

Run these from the repository root. CLIP runs on Qdrant Cloud Inference, so this dataset builds with `--inference cloud` only. The image model is a paid model, so build a small test collection first and check its token usage in the Cloud console before the full build:

```bash
uv run qdrant-datasets abo prepare
uv run qdrant-datasets abo build --inference cloud --limit 500 --collection abo-test
uv run qdrant-datasets abo build --inference cloud
uv run qdrant-datasets abo query "brown leather armchair" --mode hybrid --inference cloud
uv run qdrant-datasets abo query "https://amazon-berkeley-objects.s3.amazonaws.com/images/small/a4/a47ddc30.jpg" --image --inference cloud
uv run qdrant-datasets abo snapshot
```

`prepare` downloads the listings archive and the image index, about 90 MB, to `data/abo/raw/` the first time, and writes the products to `data/abo/records.jsonl`. `--phone-case-cap` changes how many phone cases it keeps. Qdrant Cloud Inference fetches each image from its URL during `build`, so the images themselves are never downloaded.

`query` searches with text by default. `--mode image` uses CLIP alone, `--mode text` uses BM25 alone, and `--mode hybrid` fuses both. `--image` treats the query as an image URL and finds products that look like it.

## License and Attribution

ABO is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit for the data, including the images, goes to Amazon.com. Credit for building the dataset goes to its Amazon and UC Berkeley authors. This collection uses a subset of the listings: products with an English name and a main image that is a product photo, one listing per product, one product per main image, and 1,000 of the phone cases. It adds a CLIP embedding of each main image.

> Jasmine Collins, Shubham Goel, Kenan Deng, Achleshwar Luthra, Leon Xu, Erhan Gundogdu, Xi Zhang, Tomas F. Yago Vicente, Thomas Dideriksen, Himanshu Arora, Matthieu Guillaumin, and Jitendra Malik. 2022. [ABO: Dataset and Benchmarks for Real-World 3D Object Understanding](https://arxiv.org/abs/2110.06199). In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
