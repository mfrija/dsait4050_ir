"""
Script to load and inspect the generated datasets.
Displays metadata about the first train/test pair.
"""

import os
import sys
from preprocess import load_datasets


def inspect_first_dataset():
    """
    Load the saved datasets and display metadata information about the first train/test pair.
    
    Prints:
    - Overall dataset metadata (number of variants, time period configuration)
    - First variant metadata (time point, date ranges)
    - Training set information (matrix shape, number of transactions, number of customers, assets)
    - Test set information (matrix shape, number of transactions, number of customers, assets)
    - Sample encoding information
    """
    # Construct path to processed data
    processed_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    
    print("\n" + "="*70)
    print("LOADING AND INSPECTING DATASETS")
    print("="*70)
    
    # Load datasets
    datasets = load_datasets(processed_data_path)
    
    # Get metadata
    metadata = datasets['metadata']
    
    print("\n" + "="*70)
    print("OVERALL DATASET CONFIGURATION")
    print("="*70)
    print(f"Total number of variants: {metadata['num_datasets']}")
    print(f"First time point (t0): {metadata['t0'].strftime('%Y-%m-%d')}")
    print(f"Test period: {metadata['test_period'].days} days (6 months)")
    print(f"Time step between variants: {metadata['time_step'].days} days (2 weeks)")
    print(f"Last time point: {metadata['time_points'][-1].strftime('%Y-%m-%d')}")
    
    # Inspect first dataset variant
    if len(datasets['datasets']) > 0:
        first_variant = datasets['datasets'][0]
        
        print("\n" + "="*70)
        print("FIRST DATASET VARIANT (t0)")
        print("="*70)
        print(f"Variant ID: {first_variant['variant_id']}")
        print(f"Time point (t): {first_variant['t'].strftime('%Y-%m-%d')}")
        print(f"Training period: {first_variant['training_start'].strftime('%Y-%m-%d')} to {first_variant['training_end'].strftime('%Y-%m-%d')}")
        print(f"Test period: {first_variant['test_start'].strftime('%Y-%m-%d')} to {first_variant['test_end'].strftime('%Y-%m-%d')}")
        
        # Training set information
        print("\n" + "-"*70)
        print("TRAINING SET")
        print("-"*70)
        train_matrix = first_variant['train_rel_matrix']
        train_indices = first_variant['train_indices']
        train_transactions = first_variant['train_transactions']
        
        print(f"Matrix shape (customers × assets): {train_matrix.shape}")
        print(f"  - Number of customers: {train_indices['num_customers_in_set']}")
        print(f"  - Number of assets: {train_indices['num_assets_in_set']}")
        print(f"Number of transactions: {len(train_transactions)}")
        print(f"Matrix sparsity: {(train_matrix == 0).sum() / (train_matrix.shape[0] * train_matrix.shape[1]) * 100:.2f}%")
        print(f"Total acquisition count in matrix: {train_matrix.sum():.0f}")
        
        # Sample transaction
        if len(train_transactions) > 0:
            sample_txn = train_transactions.iloc[0]
            print(f"\nSample transaction:")
            print(f"  Customer: {sample_txn['customerID']}")
            print(f"  Asset (ISIN): {sample_txn['ISIN']}")
            print(f"  Timestamp: {sample_txn['timestamp']}")
        
        # Test set information
        print("\n" + "-"*70)
        print("TEST SET")
        print("-"*70)
        test_matrix = first_variant['test_rel_matrix']
        test_indices = first_variant['test_indices']
        test_transactions = first_variant['test_transactions']
        
        print(f"Matrix shape (customers × assets): {test_matrix.shape}")
        print(f"  - Number of customers: {test_indices['num_customers_in_set']}")
        print(f"  - Number of assets: {test_indices['num_assets_in_set']}")
        print(f"Number of transactions: {len(test_transactions)}")
        print(f"Matrix sparsity: {(test_matrix == 0).sum() / (test_matrix.shape[0] * test_matrix.shape[1]) * 100:.2f}%")
        print(f"Total acquisition count in matrix: {test_matrix.sum():.0f}")
        
        # Sample transaction
        if len(test_transactions) > 0:
            sample_txn = test_transactions.iloc[0]
            print(f"\nSample transaction:")
            print(f"  Customer: {sample_txn['customerID']}")
            print(f"  Asset (ISIN): {sample_txn['ISIN']}")
            print(f"  Timestamp: {sample_txn['timestamp']}")
        
        # Encoding information
        print("\n" + "-"*70)
        print("ENCODING INFORMATION")
        print("-"*70)
        encodings = metadata.get('encodings', {})
        if encodings:
            print("Customer features encoded:")
            if 'customer_mappings' in encodings:
                for feat, mapping in encodings['customer_mappings'].items():
                    print(f"  - {feat}: {len(mapping['categories'])} categories")
            
            print("\nAsset features encoded:")
            if 'asset_mappings' in encodings:
                for feat, mapping in encodings['asset_mappings'].items():
                    print(f"  - {feat}: {len(mapping['categories'])} categories")
            
            print("\nTransaction features encoded:")
            if 'transaction_mappings' in encodings:
                for feat, mapping in encodings['transaction_mappings'].items():
                    categories_str = str(mapping['categories'][:3])  # Show first 3
                    if len(mapping['categories']) > 3:
                        categories_str = categories_str[:-1] + ", ...]"
                    print(f"  - {feat}: {len(mapping['categories'])} categories {categories_str}")
            
            print("\nMarket features encoded:")
            if 'market_mappings' in encodings:
                for feat, mapping in encodings['market_mappings'].items():
                    print(f"  - {feat}: {len(mapping['categories'])} categories")
        
        print("\n" + "="*70)
        print("✓ Dataset inspection complete")
        print("="*70 + "\n")
    else:
        print("\n✗ No datasets found in the loaded data!")


if __name__ == "__main__":
    inspect_first_dataset()
