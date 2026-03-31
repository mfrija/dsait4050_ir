"""
Neural Collaborative Filtering (NCF)

Simple MLP-based NCF:
- User embedding + Item embedding
- Concatenate → MLP → score
- Trained with binary cross-entropy

Reference: He et al. (2017) "Neural Collaborative Filtering"
"""

import numpy as np
import torch
import torch.nn as nn
from .base import BaseRecommender


# ---------------------------------------------------------------------------
# PyTorch module
# ---------------------------------------------------------------------------

class _NCFModule(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_size: int):
        super().__init__()

        self.user_embedding = nn.Embedding(n_users, embedding_size)
        self.item_embedding = nn.Embedding(n_items, embedding_size)

        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item):
        u = self.user_embedding(user)
        i = self.item_embedding(item)
        x = torch.cat([u, i], dim=1)
        return self.sigmoid(self.mlp(x)).squeeze()

    def full_sort_predict(self, users):
        """Score all items for each user."""
        user_e = self.user_embedding(users)                      # (batch, emb)
        item_e = self.item_embedding.weight                      # (n_items, emb)

        # Expand and concatenate
        user_e = user_e.unsqueeze(1).expand(-1, item_e.size(0), -1)
        item_e = item_e.unsqueeze(0).expand(user_e.size(0), -1, -1)

        x = torch.cat([user_e, item_e], dim=2)                   # (batch, n_items, 2*emb)
        scores = self.mlp(x).squeeze(-1)                         # (batch, n_items)

        return torch.sigmoid(scores)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class NCFRecommender(BaseRecommender):
    def __init__(
        self,
        embedding_size: int = 64,
        epochs: int = 10,
        batch_size: int = 2048,
        lr: float = 1e-3,
        random_state: int = 42,
    ):
        self.embedding_size = embedding_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, train_matrix):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        n_users, n_items = train_matrix.shape

        # Positive interactions
        nz = train_matrix.nonzero()
        user_ids = np.asarray(nz[0]).flatten()
        item_ids = np.asarray(nz[1]).flatten()

        # Build negatives (simple uniform sampling)
        neg_items = np.random.randint(0, n_items, size=len(user_ids))

        labels_pos = np.ones(len(user_ids))
        labels_neg = np.zeros(len(user_ids))

        users = np.concatenate([user_ids, user_ids])
        items = np.concatenate([item_ids, neg_items])
        labels = np.concatenate([labels_pos, labels_neg])

        # Shuffle
        perm = np.random.permutation(len(users))
        users, items, labels = users[perm], items[perm], labels[perm]

        self.model = _NCFModule(n_users, n_items, self.embedding_size).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCELoss()

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(users), self.batch_size):
                end = min(start + self.batch_size, len(users))

                u = torch.LongTensor(users[start:end]).to(self.device)
                i = torch.LongTensor(items[start:end]).to(self.device)
                y = torch.FloatTensor(labels[start:end]).to(self.device)

                preds = self.model(u, i)
                loss = loss_fn(preds, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    [NCF] epoch {epoch+1:3d}/{self.epochs}  loss={epoch_loss/n_batches:.4f}")

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

        with torch.no_grad():
            scores = self.model.full_sort_predict(
                torch.LongTensor(customer_indices).to(self.device)
            ).cpu().numpy()

        for i, cust_idx in enumerate(customer_indices):
            s = scores[i].copy()

            if exclude_seen:
                row = np.asarray(train_matrix[cust_idx]).flatten()
                s[row > 0] = -np.inf

            top_k_unsorted = np.argpartition(s, -k)[-k:]
            top_k = top_k_unsorted[np.argsort(s[top_k_unsorted])[::-1]]
            results[i] = top_k

        return results