"""Retrieval quality evaluation metrics.

Metrics: Recall@K, MRR (Mean Reciprocal Rank), nDCG@K.
"""
from typing import List, Set


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Calculate Recall@K.

    Args:
        retrieved: Ordered list of retrieved document IDs/contents.
        relevant: Set of relevant document IDs/contents.
        k: Number of top results to consider.

    Returns:
        Recall@K score (0-1).
    """
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & relevant) / len(relevant)


def mrr(retrieved: List[str], relevant: Set[str]) -> float:
    """Calculate Mean Reciprocal Rank.

    Args:
        retrieved: Ordered list of retrieved document IDs/contents.
        relevant: Set of relevant document IDs/contents.

    Returns:
        MRR score (0-1).
    """
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(relevances: List[float], k: int) -> float:
    """Calculate DCG@K."""
    import math
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)  # i+2 because log2(1)=0
    return dcg


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Calculate nDCG@K (Normalized Discounted Cumulative Gain).

    Binary relevance: 1 if in relevant set, 0 otherwise.
    """
    # Relevance scores for retrieved docs
    relevances = [1.0 if doc in relevant else 0.0 for doc in retrieved]
    dcg = dcg_at_k(relevances, k)

    # Ideal DCG: all relevant docs first
    ideal_relevances = [1.0] * min(len(relevant), k) + [0.0] * max(0, k - len(relevant))
    idcg = dcg_at_k(ideal_relevances, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval_batch(
    queries_results: List[dict],
    k_values: List[int] = [1, 3, 5]
) -> dict:
    """Evaluate retrieval on a batch of queries.

    Args:
        queries_results: List of dicts with keys:
            - retrieved: List[str] of retrieved docs (ordered)
            - relevant: Set[str] of relevant docs
        k_values: List of K values to evaluate.

    Returns:
        Dict with metric results.
    """
    results = {}
    for k in k_values:
        recalls = []
        ndcgs = []
        for qr in queries_results:
            recalls.append(recall_at_k(qr["retrieved"], qr["relevant"], k))
            ndcgs.append(ndcg_at_k(qr["retrieved"], qr["relevant"], k))

        results[f"Recall@{k}"] = sum(recalls) / len(recalls) if recalls else 0.0
        results[f"nDCG@{k}"] = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0

    # MRR (not K-dependent)
    mrrs = [mrr(qr["retrieved"], qr["relevant"]) for qr in queries_results]
    results["MRR"] = sum(mrrs) / len(mrrs) if mrrs else 0.0

    return results
