import numpy as np
from .base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Recommends the globally most-acquired assets."""

    def fit(self, train_matrix: np.ndarray) -> None:
        self.item_scores = np.asarray(train_matrix.sum(axis=0)).flatten().astype(np.float64)

    def recommend(
        self,
        train_matrix: np.ndarray,
        customer_indices: np.ndarray,
        k: int,
        exclude_seen: bool = True,
    ) -> np.ndarray:
        results = np.empty((len(customer_indices), k), dtype=np.int32)
        for i, cust_idx in enumerate(customer_indices):
            scores = self.item_scores.copy()
            if exclude_seen:
                row = np.asarray(train_matrix[cust_idx]).flatten()
                scores[row > 0] = -np.inf
            # argpartition gives top-k (unsorted), then sort within them
            top_k_unsorted = np.argpartition(scores, -k)[-k:]
            top_k = top_k_unsorted[np.argsort(scores[top_k_unsorted])[::-1]]
            results[i] = top_k
        return results
