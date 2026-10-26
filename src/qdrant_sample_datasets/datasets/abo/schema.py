from qdrant_client import models

from qdrant_sample_datasets.core.dataset import CollectionSpec

IMAGE_MODEL = "qdrant/clip-vit-b-32-vision"
TEXT_MODEL = "qdrant/clip-vit-b-32-text"
IMAGE_SIZE = 512
NAME_FIELD = "name"

KEYWORD = models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD)

COLLECTION = CollectionSpec(
    default_name="amazon-berkeley-objects",
    # A single unnamed vector, so queries need no `using`.
    vectors=models.VectorParams(size=IMAGE_SIZE, distance=models.Distance.COSINE),
    payload_indexes={
        "item_id": KEYWORD,
        "brand": KEYWORD,
        "color": KEYWORD,
        "product_type": KEYWORD,
        "country": KEYWORD,
        NAME_FIELD: models.TextIndexParams(
            type=models.TextIndexType.TEXT,
            tokenizer=models.TokenizerType.WORD,
            lowercase=True,
            stemmer=models.SnowballParams(
                type=models.Snowball.SNOWBALL, language=models.SnowballLanguage.ENGLISH
            ),
            stopwords=models.Language.ENGLISH,
            scoring=models.TextScoringParams(type=models.TextScoringType.BM25),
        ),
    },
)
