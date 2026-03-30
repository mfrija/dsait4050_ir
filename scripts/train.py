import os
from preprocess import load_datasets


def train():
    """
    Simple training loop that loads datasets and iterates through variants.
    
    For each dataset variant:
    - Loads training and test sets
    - Prints variant information
    - Ready for model training (placeholder)
    """
    # Construct path to processed data
    processed_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    
    print("\n" + "="*70)
    print("LOADING DATASETS FOR TRAINING")
    print("="*70)
    
    # Load datasets
    datasets = load_datasets(processed_data_path)
    
    metadata = datasets['metadata']
    num_variants = len(datasets['datasets'])
    
    print(f"\nTraining on {num_variants} dataset variants")
    print(f"Starting from t0: {metadata['t0'].strftime('%Y-%m-%d')}")
    
    print("\n" + "="*70)
    print("ITERATING THROUGH VARIANTS")
    print("="*70 + "\n")
    
    # Iterate through each variant
    for variant in datasets['datasets']:
        variant_id = variant['variant_id']
        t = variant['t']
        
        # Get train/test data
        train_rel_matrix = variant['train_rel_matrix']
        test_rel_matrix = variant['test_rel_matrix']
        train_indices = variant['train_indices']
        test_indices = variant['test_indices']
        train_transactions = variant['train_transactions']
        test_transactions = variant['test_transactions']
        
        # Print variant info
        print(f"Variant {variant_id} (t={t.strftime('%Y-%m-%d')}):")
        print(f"  Training:  {train_rel_matrix.shape} matrix, {len(train_transactions)} transactions")
        print(f"  Test:      {test_rel_matrix.shape} matrix, {len(test_transactions)} transactions")
        
        # TODO: Add model training logic here
        # - Create model
        # - Train on train_rel_matrix
        # - Evaluate on test_rel_matrix
        # - Store results
    
    print("\n" + "="*70)
    print("✓ Training complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    train()
