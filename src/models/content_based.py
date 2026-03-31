"""
Content-Based Recommender

- Uses asset features (encoded categorical columns)
- Builds user profiles as mean of interacted item features
- Recommends via cosine similarity
"""

import numpy as np
from .base import BaseRecommender


class ContentBasedRecommender(BaseRecommender):
    def __init__(self):
        self.item_features = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, train_matrix: np.ndarray, item_features: np.ndarray = None) -> None:
        """
        Parameters
        ----------
        train_matrix : (n_users, n_items)
        item_features : (n_items, n_features)
        """
        if item_features is None:
            raise ValueError("ContentBasedRecommender requires item_features")

        # Normalize item features (for cosine similarity)
        norms = np.linalg.norm(item_features, axis=1, keepdims=True) + 1e-8
        self.item_features = item_features / norms

    # ------------------------------------------------------------------
    # recommend
    # ------------------------------------------------------------------

    def recommend(
        self,
        train_matrix: np.ndarray,
        customer_indices: np.ndarray,
        k: int,
        exclude_seen: bool = True,
    ) -> np.ndarray:

        n_items = self.item_features.shape[0]
        results = np.empty((len(customer_indices), k), dtype=np.int32)

        for i, cust_idx in enumerate(customer_indices):
            user_row = np.asarray(train_matrix[cust_idx]).flatten()

            # Get interacted items
            interacted = np.where(user_row > 0)[0]

            if len(interacted) == 0:
                # fallback: recommend most "central" items (mean similarity)
                scores = self.item_features.mean(axis=0) @ self.item_features.T
            else:
                # Build user profile
                user_profile = self.item_features[interacted].mean(axis=0)

                # Normalize
                user_profile = user_profile / (np.linalg.norm(user_profile) + 1e-8)

                # Compute cosine similarity
                scores = user_profile @ self.item_features.T

            if exclude_seen:
                scores[interacted] = -np.inf

            top_k_unsorted = np.argpartition(scores, -k)[-k:]
            top_k = top_k_unsorted[np.argsort(scores[top_k_unsorted])[::-1]]

            results[i] = top_k

        return results