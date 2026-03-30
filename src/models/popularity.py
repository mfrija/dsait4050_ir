import numpy as np
import pandas as pd
from typing import Tuple, Optional


class Popularity:
    """
    Popularity-based recommendation model.
    
    Recommends assets based on their overall popularity in the training set.
    For each user, the model recommends the most popular assets that they
    haven't yet purchased.
    """
    
    def __init__(self, exclude_training_items: bool = True):
        """
        Initialize the Popularity model.
        
        Args:
            exclude_training_items: If True, exclude items the user already
                                   purchased in training from recommendations
        """
        self.exclude_training_items = exclude_training_items
        self.popularity_scores = None
        self.asset_indices = None
        self.customer_indices = None
        self.train_matrix = None
    
    def fit(self, train_rel_matrix: pd.DataFrame) -> 'Popularity':
        """
        Train the model by calculating asset popularity scores.
        
        Popularity is measured as the number of users who have purchased each asset.
        
        Args:
            train_rel_matrix: DataFrame of shape (num_customers, num_assets)
                            with binary relevance scores (0 or 1)
        
        Returns:
            self: The fitted model
        """
        self.train_matrix = train_rel_matrix.copy()
        self.customer_indices = list(train_rel_matrix.index)
        self.asset_indices = list(train_rel_matrix.columns)
        
        # Calculate popularity: sum of purchases per asset
        # (number of customers who purchased each asset)
        self.popularity_scores = train_rel_matrix.sum(axis=0)
        
        print(f"✓ Popularity model fitted")
        print(f"  Training matrix shape: {train_rel_matrix.shape}")
        print(f"  Most popular asset: {self.popularity_scores.idxmax()} "
              f"(popularity: {self.popularity_scores.max():.0f})")
        print(f"  Mean asset popularity: {self.popularity_scores.mean():.2f}")
        
        return self
    
    def predict(self, test_rel_matrix: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
        """
        Generate recommendations for all users in the test set.
        
        Args:
            test_rel_matrix: DataFrame of shape (num_customers, num_assets)
                           with binary relevance scores for test interactions
            top_k: Number of items to recommend per user
        
        Returns:
            predictions: DataFrame of shape (num_customers, top_k) with
                        recommended asset indices for each user
        """
        if self.popularity_scores is None:
            raise ValueError("Model must be fitted before making predictions")
        
        customers = test_rel_matrix.index
        predictions = {}
        
        for customer in customers:
            # Get top-k most popular assets
            top_assets = self.popularity_scores.nlargest(top_k * 2).index.tolist()
            
            # If excluding training items, filter out purchased items in training
            if self.exclude_training_items and customer in self.train_matrix.index:
                # Get items purchased by this customer in training
                purchased_items = self.train_matrix.loc[customer]
                purchased_items = set(purchased_items[purchased_items > 0].index)
                # Filter out purchased items from recommendations
                top_assets = [asset for asset in top_assets if asset not in purchased_items]
            
            # Keep only top_k
            top_assets = top_assets[:top_k]
            
            predictions[customer] = top_assets
        
        # Convert to DataFrame
        max_len = max(len(v) for v in predictions.values()) if predictions else 0
        predictions_df = pd.DataFrame.from_dict(
            predictions, 
            orient='index'
        ).iloc[:, :top_k]
        
        print(f"✓ Generated predictions for {len(predictions)} customers")
        print(f"  Recommendations per user: up to {top_k}")
        
        return predictions_df
    
    def get_recommendations(self, user_id: str, top_k: int = 10,
                           exclude_training: Optional[bool] = None) -> list:
        """
        Get top-k recommendations for a specific user.
        
        Args:
            user_id: The customer ID
            top_k: Number of items to recommend
            exclude_training: Override default exclusion behavior.
                            If None, use self.exclude_training_items
        
        Returns:
            list: Top-k recommended asset ISINs for the user
        """
        if self.popularity_scores is None:
            raise ValueError("Model must be fitted before making predictions")
        
        exclude = exclude_training if exclude_training is not None else self.exclude_training_items
        
        # Get top assets by popularity
        top_assets = self.popularity_scores.nlargest(top_k * 2).index.tolist()
        
        # Exclude training items if requested
        if exclude and user_id in self.train_matrix.index:
            purchased = self.train_matrix.loc[user_id]
            purchased = set(purchased[purchased > 0].index)
            top_assets = [asset for asset in top_assets if asset not in purchased]
        
        return top_assets[:top_k]
    
    def get_popularity_ranking(self, top_n: int = 20) -> pd.Series:
        """
        Get the top-n most popular assets.
        
        Args:
            top_n: Number of top assets to return
        
        Returns:
            pd.Series: Top assets sorted by popularity score
        """
        if self.popularity_scores is None:
            raise ValueError("Model must be fitted before retrieving rankings")
        
        return self.popularity_scores.nlargest(top_n)
