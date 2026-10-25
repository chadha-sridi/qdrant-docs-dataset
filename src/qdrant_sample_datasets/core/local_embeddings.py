from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@cache
def load_model(name: str) -> "SentenceTransformer":
    # Imported here because sentence-transformers is the optional `local` extra.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def encode(model_name: str, texts: list[str]) -> list[list[float]]:
    vectors = load_model(model_name).encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
    )
    return vectors.tolist()
