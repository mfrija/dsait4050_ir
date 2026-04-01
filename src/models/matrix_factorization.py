import numpy as np
from sklearn.decomposition import TruncatedSVD
from .base import BaseRecommender


class MFRecommender(BaseRecommender):
    """Matrix Factorization via TruncatedSVD on binarized interaction matrix."""

    def __init__(self, n_components: int = 50, n_iter: int = 10, random_state: int = 42):
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, train_matrix: np.ndarray) -> None:
        binary = (train_matrix > 0).astype(np.float32)
        n_components = min(self.n_components, min(binary.shape) - 1)
        self.svd = TruncatedSVD(
            n_components=n_components,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        latent = self.svd.fit_transform(binary)
        self.score_matrix = self.svd.inverse_transform(latent)

    def recommend(
        self,
        train_matrix: np.ndarray,
        customer_indices: np.ndarray,
        k: int,
        exclude_seen: bool = True,
    ) -> np.ndarray:
        results = np.empty((len(customer_indices), k), dtype=np.int32)
        for i, cust_idx in enumerate(customer_indices):
            scores = self.score_matrix[cust_idx].copy()
            if exclude_seen:
                row = np.asarray(train_matrix[cust_idx]).flatten()
                scores[row > 0] = -np.inf
            top_k_unsorted = np.argpartition(scores, -k)[-k:]
            top_k = top_k_unsorted[np.argsort(scores[top_k_unsorted])[::-1]]
            results[i] = top_k
        return results
