from qdrant_sample_datasets.core import text_models
from qdrant_sample_datasets.core.text_models import E5_SMALL

DENSE_MODEL = E5_SMALL
# Leave room for the "passage: " prefix and the special tokens the model adds.
TOKEN_BUDGET = 480
# Hard limit for merging tiny chunks: 512 minus 3 prefix tokens, 2 special tokens, 2 spare.
TOKEN_CEILING = 505


def count_tokens(text: str) -> int:
    return text_models.count_tokens(DENSE_MODEL, text)
