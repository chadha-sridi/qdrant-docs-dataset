from dataclasses import dataclass
from functools import cache

from qdrant_client import models
from tokenizers import Tokenizer

from qdrant_sample_datasets.core.dataset import Inference
from qdrant_sample_datasets.core.local_embeddings import encode


@dataclass(frozen=True)
class TextModel:
    name: str
    size: int
    context_window: int
    # Prefixes the model expects. Cloud Inference adds them itself, so they apply to local inference only.
    query_prefix: str = ""
    passage_prefix: str = ""


E5_SMALL = TextModel(
    name="intfloat/multilingual-e5-small",
    size=384,
    context_window=512,
    query_prefix="query: ",
    passage_prefix="passage: ",
)


@cache
def tokenizer(model_name: str) -> Tokenizer:
    return Tokenizer.from_pretrained(model_name)


def count_tokens(model: TextModel, text: str) -> int:
    return len(tokenizer(model.name).encode(text, add_special_tokens=False).ids)


def passage_vectors(
    model: TextModel, texts: list[str], inference: Inference
) -> list[list[float]] | list[models.Document]:
    if inference == "cloud":
        return [models.Document(text=text, model=model.name) for text in texts]
    return encode(model.name, [model.passage_prefix + text for text in texts])


def query_vector(
    model: TextModel, text: str, inference: Inference
) -> list[float] | models.Document:
    if inference == "cloud":
        return models.Document(text=text, model=model.name)
    return encode(model.name, [model.query_prefix + text])[0]
