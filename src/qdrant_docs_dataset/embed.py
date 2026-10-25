from functools import cache

from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer

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


@cache
def model() -> SentenceTransformer:
    return SentenceTransformer(DENSE_MODEL)


def embed_passages(texts: list[str]) -> list[list[float]]:
    return encode([f"passage: {text}" for text in texts])


def embed_query(text: str) -> list[float]:
    return encode([f"query: {text}"])[0]


def encode(texts: list[str]) -> list[list[float]]:
    vectors = model().encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
    )
    return vectors.tolist()
