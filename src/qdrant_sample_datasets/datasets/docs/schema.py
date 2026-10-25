from qdrant_client import models

from qdrant_sample_datasets.core.dataset import CollectionSpec
from qdrant_sample_datasets.datasets.docs.model import DENSE_MODEL

DENSE_VECTOR = "dense"
TEXT_FIELD = "text"

TEXT_INDEX = models.TextIndexParams(
    type=models.TextIndexType.TEXT,
    tokenizer=models.TokenizerType.CODE,
    lowercase=True,
    stemmer=models.SnowballParams(
        type=models.Snowball.SNOWBALL, language=models.SnowballLanguage.ENGLISH
    ),
    stopwords=models.Language.ENGLISH,
    scoring=models.TextScoringParams(type=models.TextScoringType.BM25),
)

COLLECTION = CollectionSpec(
    default_name="qdrant-docs",
    vectors={
        DENSE_VECTOR: models.VectorParams(
            size=DENSE_MODEL.size, distance=models.Distance.COSINE
        )
    },
    payload_indexes={
        "url": models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD),
        "content_type": models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD),
        "has_code": models.BoolIndexParams(type=models.BoolIndexType.BOOL),
        TEXT_FIELD: TEXT_INDEX,
    },
    indexing_threshold_kb=1000,
)
