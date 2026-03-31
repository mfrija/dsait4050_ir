from abc import ABC, abstractmethod
import numpy as np


class BaseRecommender(ABC):
    @abstractmethod
    def fit(self, train_matrix: np.ndarray) -> None:
        """Fit model on train_matrix (N_customers x N_items)."""
        ...

    @abstractmethod
    def recommend(
        self,
        train_matrix: np.ndarray,
        customer_indices: np.ndarray,
        k: int,
        exclude_seen: bool = True,
    ) -> np.ndarray:
        """Return top-K item indices for each customer.

        Returns
        -------
        np.ndarray of shape (len(customer_indices), k) — ranked asset column indices.
        """
        ...
