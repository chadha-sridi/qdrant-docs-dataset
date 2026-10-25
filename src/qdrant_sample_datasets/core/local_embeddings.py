from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@cache
def load_model(name: str) -> "SentenceTransformer":
    # Imported here because sentence-transformers is the optional `local` extra.
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise SystemExit(
            "--inference local needs the local extra. Run the command with `uv run --extra local`, "
            "or pass --inference cloud to let Qdrant Cloud Inference compute the embeddings."
        ) from error

    return SentenceTransformer(name)


def encode(model_name: str, texts: list[str]) -> list[list[float]]:
    vectors = load_model(model_name).encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
    )
    return vectors.tolist()
