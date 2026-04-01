"""
BPR Matrix Factorization — standalone re-implementation following RecBole's BPR model.

RecBole reference: recbole/model/general_recommender/bpr.py
  - Xavier normal initialization on embeddings
  - BPRLoss: -mean(log(sigmoid(pos_score - neg_score)))
  - Score: dot-product of user and item embeddings
  - full_sort_predict: user_emb @ all_item_emb.T
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

from .base import BaseRecommender


# ---------------------------------------------------------------------------
# RecBole-equivalent BPR loss
# ---------------------------------------------------------------------------

class BPRLoss(nn.Module):
    """Bayesian Personalised Ranking pairwise loss (mirrors RecBole's BPRLoss)."""

    def __init__(self, gamma: float = 1e-10):
        super().__init__()
        self.gamma = gamma

    def forward(self, pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
        return -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()


# ---------------------------------------------------------------------------
# Inner PyTorch module (mirrors RecBole's BPR nn.Module)
# ---------------------------------------------------------------------------

class _BPRModule(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_size: int):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_size)
        self.item_embedding = nn.Embedding(n_items, embedding_size)
        self.loss = BPRLoss()
        # Xavier normal init — same as RecBole's xavier_normal_initialization
        self.apply(self._xavier_init)

    @staticmethod
    def _xavier_init(module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def forward(self, user: torch.Tensor, item: torch.Tensor):
        return self.user_embedding(user), self.item_embedding(item)

    def calculate_loss(
        self,
        user: torch.Tensor,
        pos_item: torch.Tensor,
        neg_item: torch.Tensor,
    ) -> torch.Tensor:
        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.item_embedding(neg_item)
        pos_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_score = torch.mul(user_e, neg_e).sum(dim=1)
        return self.loss(pos_score, neg_score)

    def full_sort_predict(self, user: torch.Tensor) -> torch.Tensor:
        """Score all items for each user — mirrors RecBole's full_sort_predict."""
        user_e = self.user_embedding(user)
        all_item_e = self.item_embedding.weight          # (n_items, emb)
        return torch.matmul(user_e, all_item_e.T)        # (batch, n_items)


# ---------------------------------------------------------------------------
# BaseRecommender wrapper
# ---------------------------------------------------------------------------

class BPRRecommender(BaseRecommender):
    """BPR Matrix Factorization following RecBole's BPR model.

    Parameters
    ----------
    embedding_size : int    — latent dimension (RecBole default: 64)
    epochs         : int    — training epochs
    batch_size     : int    — mini-batch size
    lr             : float  — Adam learning rate (RecBole default: 1e-3)
    neg_samples    : int    — negatives per positive (RecBole default: 1)
    random_state   : int
    """

    def __init__(
        self,
        embedding_size: int = 64,
        epochs: int = 20,
        batch_size: int = 2048,
        lr: float = 1e-3,
        neg_samples: int = 1,
        random_state: int = 42,
    ):
        self.embedding_size = embedding_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.neg_samples = neg_samples
        self.random_state = random_state
        self.model: _BPRModule | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, train_matrix) -> None:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        n_users, n_items = train_matrix.shape

        # Extract positive (user, item) pairs — works for sparse and dense
        nz = train_matrix.nonzero()
        user_ids = np.asarray(nz[0]).flatten()
        item_ids = np.asarray(nz[1]).flatten()

        # Per-user positive item sets for O(1) negative rejection
        user_pos: dict[int, set] = {}
        for u, i in zip(user_ids, item_ids):
            user_pos.setdefault(int(u), set()).add(int(i))

        self.model = _BPRModule(n_users, n_items, self.embedding_size).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        n_interactions = len(user_ids)

        for epoch in range(self.epochs):
            self.model.train()
            perm = np.random.permutation(n_interactions)
            u_shuf = user_ids[perm]
            i_shuf = item_ids[perm]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_interactions, self.batch_size):
                end = min(start + self.batch_size, n_interactions)
                b_users = u_shuf[start:end]
                b_pos = i_shuf[start:end]
                b_neg = self._sample_negatives(b_users, user_pos, n_items)

                loss = self.model.calculate_loss(
                    torch.LongTensor(b_users).to(self.device),
                    torch.LongTensor(b_pos).to(self.device),
                    torch.LongTensor(b_neg).to(self.device),
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    [BPR] epoch {epoch+1:3d}/{self.epochs}  loss={epoch_loss/n_batches:.4f}")

    # ------------------------------------------------------------------
    # recommend
    # ------------------------------------------------------------------

    def recommend(
        self,
        train_matrix,
        customer_indices: np.ndarray,
        k: int,
        exclude_seen: bool = True,
    ) -> np.ndarray:
        self.model.eval()
        results = np.empty((len(customer_indices), k), dtype=np.int32)

        _SCORE_BATCH = 1024  # score in chunks to avoid OOM on large userbases
        all_scores = np.empty((len(customer_indices), train_matrix.shape[1]), dtype=np.float32)

        with torch.no_grad():
            for start in range(0, len(customer_indices), _SCORE_BATCH):
                end = min(start + _SCORE_BATCH, len(customer_indices))
                chunk = customer_indices[start:end]
                scores_t = self.model.full_sort_predict(
                    torch.LongTensor(chunk).to(self.device)
                )
                all_scores[start:end] = scores_t.cpu().numpy()

        for i, cust_idx in enumerate(customer_indices):
            s = all_scores[i].copy()
            if exclude_seen:
                row = np.asarray(train_matrix[cust_idx]).flatten()
                s[row > 0] = -np.inf
            top_k_unsorted = np.argpartition(s, -k)[-k:]
            top_k = top_k_unsorted[np.argsort(s[top_k_unsorted])[::-1]]
            results[i] = top_k

        return results

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_negatives(
        users: np.ndarray,
        user_pos: dict,
        n_items: int,
    ) -> np.ndarray:
        negs = np.random.randint(0, n_items, size=len(users))
        for j, u in enumerate(users):
            pos_set = user_pos.get(int(u), set())
            while negs[j] in pos_set:
                negs[j] = np.random.randint(0, n_items)
        return negs
