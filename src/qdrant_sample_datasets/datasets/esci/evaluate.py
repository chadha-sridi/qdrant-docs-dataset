import math
import random
from collections.abc import Callable

from qdrant_sample_datasets.datasets.esci.source import Query

# Gains used by the ESCI paper for nDCG: Exact, Substitute, Complement, Irrelevant.
GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
CUTOFF = 10


def sample_queries(queries: list[Query], size: int | None, seed: int) -> list[Query]:
    if size is None or size >= len(queries):
        return queries
    return random.Random(seed).sample(queries, size)


def mean_ndcg(queries: list[Query], ranked_ids: Callable[[Query], list[str]]) -> float:
    return sum(ndcg(ranked_ids(query), query.judgments) for query in queries) / len(
        queries
    )


def ndcg(ranked: list[str], judgments: dict[str, str]) -> float:
    """nDCG at the cutoff. Products without a judgment count as irrelevant."""
    gains = {product_id: GAINS[label] for product_id, label in judgments.items()}
    dcg = sum(
        gains.get(product_id, 0.0) / math.log2(rank + 2)
        for rank, product_id in enumerate(ranked[:CUTOFF])
    )
    ideal = sorted(gains.values(), reverse=True)[:CUTOFF]
    idcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(ideal))
    return dcg / idcg if idcg else 0.0
