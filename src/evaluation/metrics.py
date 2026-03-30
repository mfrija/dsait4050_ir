import numpy as np
import pandas as pd


class NDCGMetric:
    """
    Normalized Discounted Cumulative Gain at k (nDCG@k) metric for recommendation evaluation.
    
    This metric measures how close the recommendations are to the investments made by customers
    in the test period, prioritizing relevant assets in the top ranks.
    """
    
    def __init__(self, k=10):
        """
        Initialize the nDCG metric.
        
        Args:
            k (int): The number of top recommendations to consider (default: 10)
        """
        self.k = k
    
    def get_relevant_items(self, customer_transactions_df):
        """
        Extract the list of relevant items (assets acquired) from customer transaction data.
        
        An asset is considered relevant if the customer acquired (purchased) it during
        the test period (t, t + Δt).
        
        Args:
            customer_transactions_df (pd.DataFrame): DataFrame containing investment transactions 
                                                      for a single customer during the test period.
                                                      Expected columns: ['ISIN', 'transactionType', ...]
        
        Returns:
            list: List of unique ISINs (assets) that the customer acquired during the test period
                  (only Buy transactions are considered)
        """

        
        # Get unique ISINs (remove duplicates in case customer bought same asset multiple times)
        relevant_items = customer_transactions_df['ISIN'].unique().tolist()
        
        return relevant_items
    
    def compute_ndcg(self, relevant_items, recommended_items):
        """
        Compute nDCG@k score.
        
        The metric computes:
        - DCG@k: Discounted Cumulative Gain (sum of relevance values discounted by position)
        - IDCG@k: Ideal DCG@k (maximum possible DCG when all top k items are relevant)
        - nDCG@k: DCG@k / IDCG@k (normalized to range [0, 1])
        
        Formula:
            DCG@k = sum_{i=1}^{k} rel_i / log2(i+1)
            IDCG@k = sum_{i=1}^{min(k, |R|)} 1 / log2(i+1)  where |R| is number of relevant items
            nDCG@k = DCG@k / IDCG@k
        
        Args:
            relevant_items (list): List of relevant items (assets acquired by customer)
                                  Typically obtained from get_relevant_items()
            recommended_items (list): Ranked list of recommended items from the recommender model,
                                     ordered by relevance/ranking score (best to worst)
        
        Returns:
            float: nDCG@k score in range [0, 1]
                   - 1.0 means all top k recommendations match customer acquisitions
                   - 0.0 means no recommendations match customer acquisitions or no relevant items exist
        """
        # Convert to set for faster lookup
        relevant_set = set(relevant_items)
        
        # Take only the top k recommendations
        recommended_at_k = recommended_items[:self.k]
        
        # Compute DCG@k
        dcg = 0.0
        for i, item in enumerate(recommended_at_k):
            if item in relevant_set:
                # Relevance is 1 if item is in relevant set, 0 otherwise
                # Position discount: log2(i+2) because i is 0-indexed
                # Formula: rel_i / log2(position+1) where position = i+1, so log2(i+2)
                dcg += 1.0 / np.log2(i + 2)
        
        # Compute IDCG@k (ideal DCG when all top positions have relevant items)
        # This is the maximum possible DCG@k value
        idcg = 0.0
        num_relevant = min(self.k, len(relevant_items))
        for i in range(num_relevant):
            idcg += 1.0 / np.log2(i + 2)
        
        # Compute nDCG
        if idcg == 0:
            # No relevant items exist, so nDCG is 0
            return 0.0
        
        ndcg = dcg / idcg
        
        return ndcg
