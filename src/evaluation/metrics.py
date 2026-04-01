import numpy as np
from typing import Set


# ---------------------------------------------------------------------------
# Per-customer ranking metrics
# ---------------------------------------------------------------------------

def ndcg_at_k(recommendations: np.ndarray, relevant_items: Set[int], k: int) -> float:
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, item in enumerate(recommendations[:k])
        if item in relevant_items
    )
    ideal_hits = min(len(relevant_items), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# System-level beyond-accuracy metrics
# ---------------------------------------------------------------------------

def gini_index(recommendations_matrix: np.ndarray, num_items: int) -> float:
    """Gini coefficient of item recommendation frequency (0=uniform, 1=concentrated)."""
    counts = np.zeros(num_items, dtype=np.float64)
    for item in recommendations_matrix.flatten():
        counts[item] += 1
    counts_sorted = np.sort(counts)
    n = num_items
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * counts_sorted) / (n * counts_sorted.sum()) - (n + 1) / n)


def catalog_coverage(recommendations_matrix: np.ndarray, num_items: int) -> float:
    """Fraction of the catalog that appears in at least one recommendation list."""
    unique_items = np.unique(recommendations_matrix.flatten())
    return len(unique_items) / num_items


# ---------------------------------------------------------------------------
# Batch evaluator
# ---------------------------------------------------------------------------

def evaluate_model(
    recommendations: np.ndarray,
    test_matrix: np.ndarray,
    customer_indices: np.ndarray,
    num_items: int,
    k_values=(10, 20),
) -> dict:
    """Evaluate recommendations against held-out test interactions.

    Returns keys: ndcg@10, ndcg@20, gini_index, catalog_coverage.
    System-level metrics are computed at max(k_values).
    """
    max_k = max(k_values)
    per_user: dict[int, list] = {k: [] for k in k_values}

    for i, cust_idx in enumerate(customer_indices):
        relevant = set(np.where(np.asarray(test_matrix[cust_idx]).flatten() > 0)[0])
        if not relevant:
            continue
        recs = recommendations[i]
        for k in k_values:
            per_user[k].append(ndcg_at_k(recs, relevant, k))

    results = {}
    for k in k_values:
        results[f"ndcg@{k}"] = float(np.mean(per_user[k])) if per_user[k] else 0.0

    results["gini_index"] = gini_index(recommendations[:, :max_k], num_items)
    results["catalog_coverage"] = catalog_coverage(recommendations[:, :max_k], num_items)

    return results
