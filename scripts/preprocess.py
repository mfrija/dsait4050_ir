import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np
import pickle

def preprocess_data():
    """
    Read all CSV files from data/raw directory into pandas dataframes.
    Handles the FAR-Trans dataset: Financial Asset Recommendation Investment Dataset.
    
    Returns:
        dict: Dictionary containing all loaded dataframes with keys:
            - asset_information
            - close_prices
            - customer_information
            - limit_prices
            - markets
            - transactions
    """
    # Define the path to raw data directory
    raw_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    
    # Dictionary to store all dataframes
    dataframes = {}
    
    # Define CSV files with their specific dtypes and date columns
    csv_files = {
        'asset_information': {
            'filename': 'asset_information.csv',
            'dtypes': {
                'ISIN': 'str',
                'marketID': 'str'
            },
            'date_columns': ['timestamp'] # last update date
        },
        'close_prices': {
            'filename': 'close_prices.csv',
            'dtypes': {
                'ISIN': 'str',
                'closePrice': 'float'
            },
            'date_columns': ['timestamp'] # daily close price
        },
        'customer_information': {
            'filename': 'customer_information.csv',
            'dtypes': {
                'customerID': 'str', # a customer may appear multiple times given an information update
                'customerType': 'category',
                'riskLevel': 'category',
                'investmentCapacity': 'category'
            },
            'date_columns': ['timestamp', 'lastQuestionnaireDate']
        },
        'limit_prices': {
            'filename': 'limit_prices.csv',
            'dtypes': {
                'ISIN': 'str',
                'priceMinDate': 'float',
                'priceMaxDate': 'float',
                'profitability': 'float'
            },
            'date_columns': ['minDate', 'maxDate'] # starting / end dates of time series for the asset
        },
        'markets': {
            'filename': 'markets.csv',
            'dtypes': {
                'exchangeID': 'str',
                'marketID': 'str',
                'country': 'category',
                'name': 'str',
                'description': 'str',
                'country': 'str',
                'marketClass': 'str'
            },
            'date_columns': []
        },
        'transactions': {
            'filename': 'transactions.csv',
            'dtypes': {
                'customerID': 'str',
                'ISIN': 'str',
                'transactionID': 'str',
                'transactionType': 'category',
                'totalValue': 'float', # standardized and adjusted in euros
                'units': 'float',
                'channel': 'category',
                'marketID': 'str'
            },
            'date_columns': ['timestamp']
        }
    }
    
    # Load standard CSV files
    for df_name, config in csv_files.items():
        file_path = os.path.join(raw_data_path, config['filename'])
        try:
            df = pd.read_csv(file_path, dtype=config.get('dtypes', None))
            
            # Convert date columns to datetime
            for date_col in config.get('date_columns', []):
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col])
            
            dataframes[df_name] = df
            print(f"✓ Loaded {config['filename']}: {df.shape[0]:,} rows, {df.shape[1]} columns")
        except FileNotFoundError:
            print(f"✗ Warning: {config['filename']} not found at {file_path}")
        except Exception as e:
            print(f"✗ Error loading {config['filename']}: {str(e)}")
    
    # Deduplicate customer information - keep only the most recent record per customer
    print("\nProcessing customer information...")
    if 'customer_information' in dataframes:
        df_customer = dataframes['customer_information']
        initial_rows = len(df_customer)
        
        # Sort by customerID and timestamp to ensure the last record is the most recent
        df_customer = df_customer.sort_values(by=['customerID', 'timestamp'])
        
        # Keep only the last (most recent) record for each customer
        df_customer = df_customer.drop_duplicates(subset=['customerID'], keep='last')
        
        dataframes['customer_information'] = df_customer
        deduplicated_rows = len(df_customer)
        removed_rows = initial_rows - deduplicated_rows
        print(f"✓ Deduplicated customer_information: Removed {removed_rows} duplicate records, kept {deduplicated_rows} unique customers")
    
    # Deduplicate asset information - keep only the most recent record per asset
    print("\nProcessing asset information...")
    if 'asset_information' in dataframes:
        df_asset = dataframes['asset_information']
        initial_rows = len(df_asset)
        
        # Sort by ISIN and timestamp to ensure the last record is the most recent
        df_asset = df_asset.sort_values(by=['ISIN', 'timestamp'])
        
        # Keep only the last (most recent) record for each asset
        df_asset = df_asset.drop_duplicates(subset=['ISIN'], keep='last')
        
        dataframes['asset_information'] = df_asset
        deduplicated_rows = len(df_asset)
        removed_rows = initial_rows - deduplicated_rows
        print(f"✓ Deduplicated asset_information: Removed {removed_rows} duplicate records, kept {deduplicated_rows} unique assets")
    
    # Save processed dataframes to data/processed folder
    processed_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    
    # Create processed directory if it doesn't exist
    os.makedirs(processed_data_path, exist_ok=True)
    
    print("\nSaving processed dataframes...")
    for df_name, df in dataframes.items():
        try:
            output_path = os.path.join(processed_data_path, f"{df_name}.csv")
            df.to_csv(output_path, index=False)
            print(f"✓ Saved {df_name}.csv: {df.shape[0]:,} rows, {df.shape[1]} columns")
        except Exception as e:
            print(f"✗ Error saving {df_name}.csv: {str(e)}")
    
    # Generate time-based dataset variants (following paper methodology)
    print("\nGenerating the time-based dataset variants...")
    generate_datasets(dataframes, processed_data_path)
    
    print("\n✓ Preprocessing complete! Use load_datasets() to load datasets for training.")


def generate_datasets(dataframes, processed_data_path, num_variants=50):
    """
    Generate time-based dataset variants following the paper's methodology.
    
    Creates variants of the dataset, each with:
    - Training data: all transactions and prices prior to time t
    - Test data: transactions and prices in the period (t, t+Δt) where Δt = 6 months
    
    First time point t0 is August 1st, 2019.
    Time points are spaced 2 weeks apart.
    
    Constraints/Filtering rules applied:
    - If a customer acquires (buys) an asset in both training and test, keep only training
    - Keep only customers with at least one interaction in both training and test
    
    Args:
        dataframes: dict of loaded dataframes
        processed_data_path: path to data/processed folder
        num_variants: number of time-based variants to generate (default 50)
    
    Returns:
        dict: Contains metadata and lists of training/test matrices for each variant
    """
    print("\n" + "="*70)
    print("Generating dataset variants (following paper methodology)...")
    print("="*70)
    
    # Extract relevant data
    transactions = dataframes['transactions'].copy()
    asset_information = dataframes['asset_information'].copy()
    
    # Define time points
    t0 = datetime(2019, 8, 1)  # August 1st, 2019
    test_period = timedelta(days=365/2)  # 6 months
    time_step = timedelta(weeks=2)  # 2 weeks apart
    
    # Generate all time points
    time_points = [t0 + (i * time_step) for i in range(num_variants)]
    
    datasets = {
        'metadata': {
            'num_datasets': num_variants,
            't0': t0,
            'test_period': test_period,
            'time_step': time_step,
            'time_points': time_points
        },
        'datasets': []
    }
    
    print(f"\nGenerating {num_variants} datasets:")
    print(f"  First time point (t0): {t0.strftime('%Y-%m-%d')}")
    print(f"  Test period: 6 months")
    print(f"  Time step between variants: 2 weeks")
    print(f"  Last time point (t{num_variants-1}): {time_points[-1].strftime('%Y-%m-%d')}")
    
    # Get date range of data
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
    
    min_transaction_date = transactions['timestamp'].min()
    max_transaction_date = transactions['timestamp'].max()
    
    print(f"\nData availability:")
    print(f"Transactions: {min_transaction_date.strftime('%Y-%m-%d')} to {max_transaction_date.strftime('%Y-%m-%d')}")
    
    # Generate variants
    for idx, t_current in enumerate(time_points):
        # Use all available data prior to the time point for training
        training_start = min_transaction_date
        training_end = t_current
        test_end = t_current + test_period
        
        # Skip if test period exceeds available data from transaction data
        if test_end > max_transaction_date:
            print(f"\n  ⊘ Dataset t{idx} ({t_current.strftime('%Y-%m-%d')}): Test period exceeds available data, skipping")
            continue
        
        # Get training investment transactions (before t_current) - only Buy transactions
        train_transactions = transactions[
            (transactions['timestamp'] >= training_start) & 
            (transactions['timestamp'] < training_end) &
            (transactions['transactionType'] == 'Buy')
        ].copy()
        
        # Get test investment transactions (in period (t_current, t_current + 6 months)) - only Buy transactions
        test_transactions = transactions[
            (transactions['timestamp'] >= training_end) & 
            (transactions['timestamp'] < test_end) &
            (transactions['transactionType'] == 'Buy')
        ].copy()
        
        # Apply filtering rules        
        train_customers = set(train_transactions['customerID'].unique())
        test_customers = set(test_transactions['customerID'].unique())

        # Keep only customers with interactions in both training and test sets
        kept_customers = train_customers & test_customers
        
        train_transactions = train_transactions[train_transactions["customerID"].isin(kept_customers)].copy()
        test_transactions = test_transactions[test_transactions["customerID"].isin(kept_customers)].copy()

        # Remove customer-asset pairs that appear in both training and test
        # Keep only training interactions
        # Goal of the trained recommender model is to suggest other assets than those acquired during the training time interval
        train_pairs = set(zip(train_transactions['customerID'], train_transactions['ISIN']))
        test_transactions = test_transactions[
            ~test_transactions.apply(lambda row: (row['customerID'], row['ISIN']) in train_pairs, axis=1)
        ].copy()
        
        # Create binary interaction matrices for the train and test sets
        # For each amtrix consider all the spectrum of avaialble assets
        all_assets = sorted(list(asset_information["ISIN"].unique()))
        
        # Build binary interaction matrix for the training transactions
        train_rel_matrix = build_rel_matrix(train_transactions, all_assets=all_assets)
        
        # Build binary interaction matrix for the testing transactions
        test_rel_matrix = build_rel_matrix(test_transactions, all_assets=all_assets)
        
        # Store variant information
        variant_info = {
            'variant_id': idx,
            't': t_current,
            'training_start': training_start,
            'training_end': training_end,
            'test_start': training_end,
            'test_end': test_end,
            'train_transactions': train_transactions,
            'test_transactions': test_transactions,
            'train_rel_matrix': train_rel_matrix,
            'test_rel_matrix': test_rel_matrix
        }
        
        datasets['datasets'].append(variant_info)
        
        print(f"✓ Generated variant t{idx} ({t_current.strftime('%Y-%m-%d')}): "
                f"{len(train_transactions)} train transactions, {len(test_transactions)} test transactions, "
                f"Train Rel matrix dims: {train_rel_matrix.shape} "
                f"Test Rel matrix dims: {test_rel_matrix.shape}")
    
    print(f"\n✓ Successfully generated {len(datasets['datasets'])} dataset variants")
    
    # Save datasets to pickle file
    save_datasets(datasets, processed_data_path)
    
    return datasets

def build_rel_matrix(transactions, all_assets):
    """
    Build a full relevance matrix (Rel) with union dimensions.
    
    Args:
        transactions: Transactions DataFrame with columns [customerID, ISIN]
        all_customers: list of all unique customer IDs (with at least 1 transaction in the train and test sets)
        all_assets: list of all available unique asset ISINs
    
    Returns:
        pandas DataFrame: Rel matrix with shape (len(all_customers), len(all_assets))
                        where Rel[u, i] = 1.0 if customer u bought asset i, else 0.0
    """
    # Create dictionary for fast lookup of indices
    customers = sorted(list(transactions["customerID"].unique()))

    customer_idx = {cid: i for i, cid in enumerate(customers)}
    asset_idx = {isin: j for j, isin in enumerate(all_assets)}
    
    # Initialize matrix with zeros
    rel_matrix = np.zeros((len(customers), len(all_assets)), dtype=np.float32)
    
    # Fill in interactions
    for _, row in transactions.iterrows():
        if row['customerID'] in customer_idx and row['ISIN'] in asset_idx:
            i = customer_idx[row['customerID']]
            j = asset_idx[row['ISIN']]
            rel_matrix[i, j] = 1.0
    
    # Create DataFrame with indices
    rel_df = pd.DataFrame(
        rel_matrix,
        index=customers,
        columns=all_assets
    )
    return rel_df

def save_datasets(datasets, processed_data_path):
    """
    Save the datasets object to a pickle file.
    
    Args:
        datasets: dict containing all dataset variants
        processed_data_path: path to data/processed folder
    """
    datasets_file = os.path.join(processed_data_path, 'datasets.pkl')
    
    print("\nSaving datasets to pickle file...")
    with open(datasets_file, 'wb') as f:
        pickle.dump(datasets, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size_mb = os.path.getsize(datasets_file) / (1024**2)
    print(f"✓ Saved {len(datasets['datasets'])} dataset variants to {datasets_file}")
    print(f"  File size: {file_size_mb:.2f} MB")

def load_datasets(processed_data_path):
    """
    Load all datasets from pickle file.
    
    Args:
        processed_data_path: path to data/processed folder
    
    Returns:
        dict: Complete datasets object with metadata and all variants
    """    
    datasets_file = os.path.join(processed_data_path, 'datasets.pkl')
    
    print(f"\nLoading datasets from {datasets_file}...")
    with open(datasets_file, 'rb') as f:
        datasets = pickle.load(f)
    
    print(f"✓ Loaded {len(datasets['datasets'])} dataset variants")
    print(f"  Metadata: t0={datasets['metadata']['t0'].strftime('%Y-%m-%d')}, "
          f"num_datasets={datasets['metadata']['num_datasets']}")
    
    return datasets

if __name__ == "__main__":
    preprocess_data()