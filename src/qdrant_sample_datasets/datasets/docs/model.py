from functools import cache

from tokenizers import Tokenizer

from qdrant_sample_datasets.core.local_embeddings import encode

DENSE_MODEL = "intfloat/multilingual-e5-small"
DENSE_SIZE = 384
CONTEXT_WINDOW = 512
# Leave room for the "passage: " prefix and the special tokens the model adds.
TOKEN_BUDGET = 480
# Hard limit for merging tiny chunks: 512 minus 3 prefix tokens, 2 special tokens, 2 spare.
TOKEN_CEILING = 505


@cache
def tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained(DENSE_MODEL)


def count_tokens(text: str) -> int:
    return len(tokenizer().encode(text, add_special_tokens=False).ids)


def embed_passages(texts: list[str]) -> list[list[float]]:
    return encode(DENSE_MODEL, [f"passage: {text}" for text in texts])


def embed_query(text: str) -> list[float]:
    return encode(DENSE_MODEL, [f"query: {text}"])[0]
