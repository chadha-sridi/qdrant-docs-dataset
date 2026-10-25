from qdrant_client import models

from qdrant_sample_datasets.core.dataset import CollectionSpec
from qdrant_sample_datasets.core.text_models import E5_SMALL

DENSE_MODEL = E5_SMALL
DENSE_VECTOR = "dense"
TEXT_FIELD = "text"

TEXT_INDEX = models.TextIndexParams(
    type=models.TextIndexType.TEXT,
    tokenizer=models.TokenizerType.WORD,
    lowercase=True,
    stemmer=models.SnowballParams(
        type=models.Snowball.SNOWBALL, language=models.SnowballLanguage.ENGLISH
    ),
    stopwords=models.Language.ENGLISH,
    scoring=models.TextScoringParams(type=models.TextScoringType.BM25),
)

COLLECTION = CollectionSpec(
    default_name="esci-products",
    vectors={
        DENSE_VECTOR: models.VectorParams(
            size=DENSE_MODEL.size, distance=models.Distance.COSINE
        )
    },
    payload_indexes={
        "product_id": models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD),
        "brand": models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD),
        "color": models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD),
        TEXT_FIELD: TEXT_INDEX,
    },
)
