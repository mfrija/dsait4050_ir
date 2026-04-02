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

def risk_fit_at_k(recommendations_risk_scores: np.ndarray, customer_risk_score: float, k: int) -> float:
    risk_fit = np.sum(
        [(1.0/np.log2(rank+2))*(1.0-(r_a - customer_risk_score)/3.0) if (r_a > customer_risk_score) else (1.0/np.log2(rank+2)) for rank, r_a in enumerate(recommendations_risk_scores[:k])]
    )
    normalization_factor = np.sum([(1.0/np.log2(rank+2)) for rank, _ in enumerate(recommendations_risk_scores[:k])])
    risk_fit = risk_fit/normalization_factor
    assert risk_fit <= 1.0 , f"Risk Fit: {risk_fit}"
    return risk_fit

def rec_popularity_at_k(recommendations_popularity_scores: np.ndarray, k: int) -> float:
    return np.sum(recommendations_popularity_scores[:k])/k
    

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
    customer_risk_scores: dict,
    asset_risk_scores: dict,
    asset_popularity: dict,
    num_items: int,
    k_values=(10, 20)
) -> dict:
    """Evaluate recommendations against held-out test interactions.

    Returns keys: ndcg@10, ndcg@20, risk_fit@10, risk_fit@20, gini_index, catalog_coverage.
    System-level metrics are computed at max(k_values).
    """
    max_k = max(k_values)
    per_user_ndcg_at_k: dict[int, list] = {k: [] for k in k_values}
    per_user_risk_fit_at_k: dict[int, list] = {k: [] for k in k_values}
    per_user_rec_pop_at_k: dict[int, list] = {k: [] for k in k_values}

    for i, cust_idx in enumerate(customer_indices):
        relevant = set(np.where(np.asarray(test_matrix[cust_idx]).flatten() > 0)[0])
        cust_risk_score = customer_risk_scores[cust_idx]
        if not relevant:
            continue
        recs = recommendations[i]
        recs_risk_scores = np.asarray([asset_risk_scores[int(idx)] for idx in recs])
        recs_pop_scores = np.asanyarray([asset_popularity[int(idx)] for idx in recs])
        for k in k_values:
            per_user_ndcg_at_k[k].append(ndcg_at_k(recommendations=recs, 
                                                   relevant_items=relevant,
                                                   k=k))
            per_user_risk_fit_at_k[k].append(risk_fit_at_k(recommendations_risk_scores=recs_risk_scores, 
                                                           customer_risk_score=cust_risk_score, 
                                                           k=k))
            per_user_rec_pop_at_k[k].append(rec_popularity_at_k(recommendations_popularity_scores=recs_pop_scores,
                                                                k=k))

    results = {}
    for k in k_values:
        results[f"ndcg@{k}"] = float(np.mean(per_user_ndcg_at_k[k])) if per_user_ndcg_at_k[k] else 0.0
        results[f"risk_fit@{k}"] = float(np.mean(per_user_risk_fit_at_k[k])) if per_user_risk_fit_at_k[k] else 0.0
        results[f"rec_pop@{k}"] = float(np.mean(per_user_rec_pop_at_k[k])) if per_user_rec_pop_at_k[k] else 0.0

    results["gini_index"] = gini_index(recommendations[:, :max_k], num_items)
    results["catalog_coverage"] = catalog_coverage(recommendations[:, :max_k], num_items)

    return results
