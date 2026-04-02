import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder


def encode_categorical_features(dataframes):
    """
    Encode categorical features in customer, asset, transaction, and market dataframes.
    
    Categorical features encoded:
    - Customer: customerType, riskLevel, investmentCapacity
    - Asset: assetCategory, assetSubCategory, marketID, sector, industry
    - Transaction: transactionType, channel
    - Market: country, marketClass
    
    Note: NaN values are replaced with "Unknown" category before encoding to ensure
    all rows can be encoded without missing values.
    
    Args:
        dataframes: dict of loaded dataframes
    
    Returns:
        tuple: (dataframes_updated, encodings_dict) where:
            - dataframes_updated: modified dataframes with encoded categorical features
            - encodings_dict: dict containing all encoders and their mappings for reversal
    """
    encodings = {
        'customer_encoders': {},
        'asset_encoders': {},
        'transaction_encoders': {},
        'market_encoders': {},
        'customer_mappings': {},
        'asset_mappings': {},
        'transaction_mappings': {},
        'market_mappings': {}
    }
    
    # Encode customer categorical features
    print("\nEncoding customer categorical features...")
    if 'customer_information' in dataframes:
        df_customer = dataframes['customer_information'].copy()
        customer_categorical_cols = ['customerType', 'riskLevel', 'investmentCapacity']
        
        for col in customer_categorical_cols:
            if col in df_customer.columns:
                # Convert categorical dtype to object (string) to allow adding 'Unknown' category
                df_customer[col] = df_customer[col].astype('object')
                
                # Replace NaN with "Unknown" category
                df_customer[col] = df_customer[col].fillna('Unknown')
                
                encoder = LabelEncoder()
                # Fit the encoder on all values (including "Unknown")
                unique_values = df_customer[col].unique()
                encoder.fit(unique_values)
                
                # Store encoder and mapping for reversal
                encodings['customer_encoders'][col] = encoder
                encodings['customer_mappings'][col] = {
                    'categories': list(encoder.classes_),
                    'encoded_values': list(range(len(encoder.classes_)))
                }
                
                # Apply encoding
                df_customer[col] = encoder.transform(df_customer[col])
                print(f"  ✓ Encoded {col}: {len(encoder.classes_)} unique classes (including 'Unknown' for NaN)")
        
        dataframes['customer_information'] = df_customer
    
    # Encode asset categorical features
    print("\nEncoding asset categorical features...")
    if 'asset_information' in dataframes:
        df_asset = dataframes['asset_information'].copy()
        asset_categorical_cols = ['assetCategory', 'assetSubCategory', 'sector', 'industry']
        
        for col in asset_categorical_cols:
            if col in df_asset.columns:
                # Convert categorical dtype to object (string) to allow adding 'Unknown' category
                df_asset[col] = df_asset[col].astype('object')
                
                # Replace NaN with "Unknown" category
                df_asset[col] = df_asset[col].fillna('Unknown')
                
                encoder = LabelEncoder()
                # Fit the encoder on all values (including "Unknown")
                unique_values = df_asset[col].unique()
                encoder.fit(unique_values)
                
                # Store encoder and mapping for reversal
                encodings['asset_encoders'][col] = encoder
                encodings['asset_mappings'][col] = {
                    'categories': list(encoder.classes_),
                    'encoded_values': list(range(len(encoder.classes_)))
                }
                
                # Apply encoding
                df_asset[col] = encoder.transform(df_asset[col])
                print(f"  ✓ Encoded {col}: {len(encoder.classes_)} unique classes (including 'Unknown' for NaN)")
        
        dataframes['asset_information'] = df_asset
    
    # Encode transaction categorical features
    print("\nEncoding transaction categorical features...")
    if 'transactions' in dataframes:
        df_transactions = dataframes['transactions'].copy()
        transaction_categorical_cols = ['transactionType', 'channel']
        
        for col in transaction_categorical_cols:
            if col in df_transactions.columns:
                # Convert categorical dtype to object (string) to allow adding 'Unknown' category
                df_transactions[col] = df_transactions[col].astype('object')
                
                # Replace NaN with "Unknown" category
                df_transactions[col] = df_transactions[col].fillna('Unknown')
                
                encoder = LabelEncoder()
                # Fit the encoder on all values (including "Unknown")
                unique_values = df_transactions[col].unique()
                encoder.fit(unique_values)
                
                # Store encoder and mapping for reversal
                encodings['transaction_encoders'][col] = encoder
                encodings['transaction_mappings'][col] = {
                    'categories': list(encoder.classes_),
                    'encoded_values': list(range(len(encoder.classes_)))
                }
                
                # Apply encoding
                df_transactions[col] = encoder.transform(df_transactions[col])
                print(f"  ✓ Encoded {col}: {len(encoder.classes_)} unique classes (including 'Unknown' for NaN)")
        
        dataframes['transactions'] = df_transactions
    
    # Encode market categorical features
    print("\nEncoding market categorical features...")
    if 'markets' in dataframes:
        df_markets = dataframes['markets'].copy()
        market_categorical_cols = ['country', 'marketClass']
        
        for col in market_categorical_cols:
            if col in df_markets.columns:
                # Convert categorical dtype to object (string) to allow adding 'Unknown' category
                df_markets[col] = df_markets[col].astype('object')
                
                # Replace NaN with "Unknown" category
                df_markets[col] = df_markets[col].fillna('Unknown')
                
                encoder = LabelEncoder()
                # Fit the encoder on all values (including "Unknown")
                unique_values = df_markets[col].unique()
                encoder.fit(unique_values)
                
                # Store encoder and mapping for reversal
                encodings['market_encoders'][col] = encoder
                encodings['market_mappings'][col] = {
                    'categories': list(encoder.classes_),
                    'encoded_values': list(range(len(encoder.classes_)))
                }
                
                # Apply encoding
                df_markets[col] = encoder.transform(df_markets[col])
                print(f"  ✓ Encoded {col}: {len(encoder.classes_)} unique classes (including 'Unknown' for NaN)")
        
        dataframes['markets'] = df_markets
    
    print("✓ Categorical feature encoding complete")
    return dataframes, encodings


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
    
    # Generate time-based dataset variants (following paper methodology)
    print("\nPreprocessing data...")
    
    # Encode categorical features before generating datasets
    print("\n" + "="*70)
    print("STEP 1: Encoding categorical features...")
    print("="*70)
    dataframes, encodings = encode_categorical_features(dataframes)
    
    print("\n" + "="*70)
    print("\nSaving processed dataframes...")
    print("="*70)
    for df_name, df in dataframes.items():
        try:
            output_path = os.path.join(processed_data_path, f"{df_name}.csv")
            df.to_csv(output_path, index=False)
            print(f"✓ Saved {df_name}.csv: {df.shape[0]:,} rows, {df.shape[1]} columns")
        except Exception as e:
            print(f"✗ Error saving {df_name}.csv: {str(e)}")
    
    # Generate dataset variants with encoding information
    print("\n" + "="*70)
    print("STEP 2: Generating time-based dataset variants...")
    print("="*70)
    generate_datasets(dataframes, processed_data_path, encodings)
    
    print("\n✓ Preprocessing complete! Use load_datasets() to load datasets for training.")

def generate_datasets(dataframes, processed_data_path, encodings, num_variants=30):
    """
    Generate time-based dataset variants following the FAR-Trans paper's methodology.
    
    Creates variants of the dataset, each with:
    - Training data: all transactions and prices prior to time t
    - Test data: transactions and prices in the period (t, t+Δt) where Δt = 6 months
    
    First time point t0 is August 1st, 2019.
    Time points are spaced 2 weeks apart.
    
    All relevance matrices use globally-consistent customer and asset indices across variants,
    ensuring consistency for model training and evaluation.
    
    Constraints/Filtering rules applied:
    - If a customer acquires (buys) an asset in both training and test, keep only training
    - Keep only customers with at least one interaction in both training and test
    
    Args:
        dataframes: dict of loaded dataframes with encoded features
        processed_data_path: path to data/processed folder
        num_variants: number of time-based variants to generate (default 50)
    
    Returns:
        dict: Contains metadata (with all encodings) and datasets with globally-indexed matrices
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
            'time_points': time_points,
            'encodings': encodings  # Store categorical feature encodings with datasets
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

    # The spectrum of available assets
    all_assets = sorted(list(asset_information["ISIN"].unique()))
    
    min_transaction_date = transactions['timestamp'].min()
    max_transaction_date = transactions['timestamp'].max()
    
    print(f"\nData availability:")
    print(f"Transactions: {min_transaction_date.strftime('%Y-%m-%d')} to {max_transaction_date.strftime('%Y-%m-%d')}")
    
    # Get the encoded value for 'Buy' transaction type
    buy_encoded = None
    if 'transaction_mappings' in encodings and 'transactionType' in encodings['transaction_mappings']:
        categories = encodings['transaction_mappings']['transactionType']['categories']
        if 'Buy' in categories:
            buy_encoded = categories.index('Buy')
    
    if buy_encoded is None:
        raise ValueError("Could not find encoded value for 'Buy' transactionType. Check encodings.")
    
    print(f"Using encoded value {buy_encoded} for 'Buy' transactions")
    
    # Create customer_id_to_risk_level mapping at the beginning of dataset generation
    # This decodes encoded risk levels once instead of repeatedly for each matrix
    print(f"\nMapping customer risk levels...")
    customer_id_to_risk_score = {}
    
    # Define risk level mapping scheme - maps categorical values to numerical scores
    categorical_to_score = {
        'Conservative': 1.0,
        'Predicted_Conservative': 1.0,
        'Income': 2.0,
        'Predicted_Income': 2.0,
        'Not_Available': 2.0,
        'Balanced': 3.0,
        'Predicted_Balanced': 3.0,
        'Aggressive': 4.0,
        'Predicted_Aggressive': 4.0
    }
    
    # Get the riskLevel encoder from encodings
    risk_encoder = None
    if 'customer_encoders' in encodings and 'riskLevel' in encodings['customer_encoders']:
        risk_encoder = encodings['customer_encoders']['riskLevel']
    
    # Decode and map all customer risk levels
    for _, row in dataframes['customer_information'].iterrows():
        customer_id = row['customerID']
        encoded_risk = row['riskLevel']
        
        # Decode the encoded integer value back to categorical string
        categorical_risk = 'Unknown'  # Default
        if risk_encoder is not None:
            try:
                categorical_risk = risk_encoder.inverse_transform([int(encoded_risk)])[0]
            except Exception:
                categorical_risk = 'Unknown'
        else:
            # If no encoder, assume the value is already categorical
            categorical_risk = str(encoded_risk)
        
        # Map categorical risk to numerical score
        risk_score = categorical_to_score.get(categorical_risk, 2.0)  # Default to 2.0 (Income/Unknown)
        customer_id_to_risk_score[customer_id] = risk_score
    
    print(f"✓ Mapped risk levels for {len(customer_id_to_risk_score)} customers")
    
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
            (transactions['transactionType'] == buy_encoded)
        ].copy()
        
        # Get test investment transactions (in period (t_current, t_current + 6 months)) - only Buy transactions
        test_transactions = transactions[
            (transactions['timestamp'] >= training_end) & 
            (transactions['timestamp'] < test_end) &
            (transactions['transactionType'] == buy_encoded)
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
            
        # Build count-based interaction matrix for the training transactions
        train_rel_matrix, train_indices = build_rel_matrix(
            train_transactions, customers=kept_customers, all_assets=all_assets, 
            customer_risk_scores=customer_id_to_risk_score,
            get_asset_risk_scores=True
        )
        
        # Build count-based interaction matrix for the testing transactions
        test_rel_matrix, test_indices = build_rel_matrix(
            test_transactions, customers=kept_customers, all_assets=all_assets,
            customer_risk_scores=customer_id_to_risk_score
        )
        
        # Print a sample row with non-zero values from training matrix
        train_sample_printed = False
        # Get indices of rows with non-zero values
        train_nonzero_rows = [i for i in range(train_rel_matrix.shape[0]) if np.any(train_rel_matrix[i, :] > 0)]
        if train_nonzero_rows:
            # Pick a random row from those with non-zero values
            i = np.random.choice(train_nonzero_rows)
            customer_id = train_indices['customer_local_idx_to_id'][i]
            nonzero_indices = np.nonzero(train_rel_matrix[i, :])[0]
            asset_values = [(train_indices['asset_local_idx_to_id'][j], train_rel_matrix[i, j]) for j in nonzero_indices]
            print(f"  Sample training row - Customer {customer_id}: {asset_values[:3]}")  # Print first 3 assets
            # Print risk score for this customer
            risk_score = train_indices['customer_local_idx_to_risk_score'].get(i, 'Unknown')
            print(f"    Customer Risk Score: {risk_score}")
            # Print proxy risk scores for assets purchased by this customer
            asset_risk_scores = [train_indices['asset_local_idx_to_risk_score'].get(j, 'Unknown') for j in nonzero_indices[:3]]
            print(f"    Asset Proxy Risk Scores: {asset_risk_scores}")
            train_sample_printed = True
        
        # Print size of asset_local_idx_to_risk_score mapping
        asset_risk_score_size = len(train_indices['asset_local_idx_to_risk_score'])
        print(f"  Asset Risk Score Index Size: {asset_risk_score_size}")
        
        # Print a sample row with non-zero values from test matrix
        test_sample_printed = False
        # Get indices of rows with non-zero values
        test_nonzero_rows = [i for i in range(test_rel_matrix.shape[0]) if np.any(test_rel_matrix[i, :] > 0)]
        if test_nonzero_rows:
            # Pick a random row from those with non-zero values
            i = np.random.choice(test_nonzero_rows)
            customer_id = test_indices['customer_local_idx_to_id'][i]
            nonzero_indices = np.nonzero(test_rel_matrix[i, :])[0]
            asset_values = [(test_indices['asset_local_idx_to_id'][j], test_rel_matrix[i, j]) for j in nonzero_indices]
            print(f"  Sample test row - Customer {customer_id}: {asset_values[:3]}")  # Print first 3 assets
            # Print risk score for this customer
            risk_score = test_indices['customer_local_idx_to_risk_score'].get(i, 'Unknown')
            print(f"    Customer Risk Score: {risk_score}")
            # Print proxy risk scores for assets purchased by this customer
            asset_risk_scores = [test_indices['asset_local_idx_to_risk_score'].get(j, 'Unknown') for j in nonzero_indices[:3]]
            print(f"    Asset Proxy Risk Scores: {asset_risk_scores}")
            test_sample_printed = True
        
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
            'test_rel_matrix': test_rel_matrix,
            'train_indices': train_indices,  # Customer and asset index mappings for training
            'test_indices': test_indices      # Customer and asset index mappings for testing
        }
        
        datasets['datasets'].append(variant_info)
        
        print(f"✓ Generated variant t{idx} ({t_current.strftime('%Y-%m-%d')}): "
                f"{len(train_transactions)} train transactions, {len(test_transactions)} test transactions, "
                f"Train matrix dims: {train_rel_matrix.shape} (count-based), "
                f"Test matrix dims: {test_rel_matrix.shape} (count-based)")
    
    print(f"\n✓ Successfully generated {len(datasets['datasets'])} dataset variants")
    
    # Save datasets to pickle file
    save_datasets(datasets, processed_data_path)
    
    return datasets

def build_rel_matrix(transactions, customers, all_assets, customer_risk_scores=None, get_asset_risk_scores=False):
    """
    Build a count-based relevance matrix (Rel) with local indexing for both customers and assets.
    
    The relevance matrix stores counts instead of binary values, reflecting the number of times
    each customer acquired each asset.
    
    - Customer rows: Local indices for customers in this transaction set
    - Asset columns: Local indices for assets in this transaction set
    
    Args:
        transactions: Transactions DataFrame with columns [customerID, ISIN]
        customers: set of customer IDs to include
        all_assets: list of all available unique asset ISINs
        customer_risk_scores: Optional dict mapping {customerID -> risk_score} from pre-computed mapping
    
    Returns:
        tuple: (rel_matrix, indices_dict) where:
            - rel_matrix: numpy array with shape (num_customers_in_set, num_assets_in_set)
                         containing counts of acquisitions (dtype: float32)
                         - rows: local indices for customers in this set
                         - cols: local indices for assets in this set
            - indices_dict: dict containing the mappings:
                - 'customer_id_to_local_idx': Local {customerID -> local row index}
                - 'customer_local_idx_to_id': Local {local row index -> customerID}
                - 'asset_id_to_local_idx': Local {assetISIN -> local column index}
                - 'asset_local_idx_to_id': Local {local column index -> assetISIN}
                - 'num_customers_in_set': Number of customers in this set
                - 'num_assets_in_set': Number of assets in this set
                - 'customer_local_idx_to_risk_score': Mapping {local customer index -> risk_score}
                - 'asset_local_idx_to_risk_score': Proxy risk score for each asset {local asset index -> proxy_risk_score}
    """
    # The customers with transactions both in the training and test transactions subsets
    customers_in_set = sorted(list(customers))
    num_customers = len(customers_in_set)
    
    # Create LOCAL customer index mappings
    customer_id_to_local_idx = {cid: i for i, cid in enumerate(customers_in_set)}
    customer_local_idx_to_id = {i: cid for i, cid in enumerate(customers_in_set)}
    
    # Create customer_id_to_risk_level mapping using pre-computed mapping if provided
    customer_local_idx_to_risk_score = {}
    if customer_risk_scores is not None:
        for customer_id in customers_in_set:
            if customer_id in customer_risk_scores:
                risk_score = customer_risk_scores[customer_id]
                # Create mapping from local index to risk score
                local_idx = customer_id_to_local_idx[customer_id]
                customer_local_idx_to_risk_score[local_idx] = risk_score
    
    # Get the number of all available assets (sorted for consistency)
    num_assets = len(all_assets)
    
    # Create LOCAL asset index mappings
    asset_id_to_local_idx = {asset_id: j for j, asset_id in enumerate(all_assets)}
    asset_local_idx_to_id = {j: asset_id for j, asset_id in enumerate(all_assets)}
    
    # Initialize matrix: rows = customers in this set, columns = assets in this set
    rel_matrix = np.zeros((num_customers, num_assets), dtype=np.float32)
    
    # Fill in interaction counts using local indices
    for _, row in transactions.iterrows():
        customer_id = row['customerID']
        asset_id = row['ISIN']
        
        if customer_id in customer_id_to_local_idx and asset_id in asset_id_to_local_idx:
            local_i = customer_id_to_local_idx[customer_id]  # Local row index
            local_j = asset_id_to_local_idx[asset_id]        # Local column index
            rel_matrix[local_i, local_j] += 1.0  # Increment count
    
    # Compute proxy risk score for each asset
    asset_local_idx_to_risk_score = {}
    if get_asset_risk_scores:
        if len(customer_local_idx_to_risk_score) > 0:
            # Risk score values used for weighting
            risk_score_values = [1.0, 2.0, 3.0, 4.0]
            
            # For each asset (column in the matrix)
            for asset_local_idx in list(asset_local_idx_to_id.keys()):
                # Find customers who bought this asset (non-zero entries in this column)
                customers_who_bought = np.nonzero(rel_matrix[:, asset_local_idx])[0]
                
                if len(customers_who_bought) > 0:
                    # Get risk scores for these customers
                    risk_scores_for_asset = [
                        customer_local_idx_to_risk_score.get(customer_local_idx, 2.0) 
                        for customer_local_idx in customers_who_bought
                    ]
                    
                    # Count proportion of each risk score
                    total_customers = len(risk_scores_for_asset)
                    proportions = {}
                    for score in risk_score_values:
                        count = sum(1 for rs in risk_scores_for_asset if rs == score)
                        proportions[score] = count / total_customers
                    
                    # Compute weighted sum as proxy risk score
                    proxy_risk = sum(score * proportions[score] for score in risk_score_values)
                    asset_local_idx_to_risk_score[asset_local_idx] = proxy_risk
                else:
                    # We assume a neutral risk profile for assets without any past acquisitions
                    proxy_risk = 2.0
                    asset_local_idx_to_risk_score[asset_local_idx] = proxy_risk
    
    # Store index information
    indices_dict = {
        'customer_id_to_local_idx': customer_id_to_local_idx,
        'customer_local_idx_to_id': customer_local_idx_to_id,
        'customer_local_idx_to_risk_score': customer_local_idx_to_risk_score,
        'asset_id_to_local_idx': asset_id_to_local_idx,
        'asset_local_idx_to_id': asset_local_idx_to_id,
        'asset_local_idx_to_risk_score': asset_local_idx_to_risk_score,
        'num_customers_in_set': num_customers,
        'num_assets_in_set': num_assets,
    }
    
    return rel_matrix, indices_dict

def save_datasets(datasets, processed_data_path):
    """
    Save the datasets object to a pickle file.
    
    The datasets object includes all categorical feature encodings in its metadata,
    so they are saved together with the datasets.
    
    Args:
        datasets: dict containing all dataset variants and encodings
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
    
    The datasets object includes all categorical feature encodings in its metadata,
    which can be used to:
    - Reverse encode categorical features back to original values
    - Understand the feature encoding during model training
    
    Args:
        processed_data_path: path to data/processed folder
    
    Returns:
        dict: Complete datasets object with metadata (including encodings) and all variants
    """    
    datasets_file = os.path.join(processed_data_path, 'datasets.pkl')
    
    print(f"\nLoading datasets from {datasets_file}...")
    with open(datasets_file, 'rb') as f:
        datasets = pickle.load(f)
    
    print(f"✓ Loaded {len(datasets['datasets'])} dataset variants")
    print(f"  Metadata: t0={datasets['metadata']['t0'].strftime('%Y-%m-%d')}, "
          f"num_datasets={datasets['metadata']['num_datasets']}")
    
    return datasets

def decode_categorical_value(encoded_value, feature_name, encodings):
    """
    Helper function to decode a categorical encoded value back to its original string.
    
    Supports all 4 encoding types: customer, asset, transaction, and market features.
    
    Args:
        encoded_value: integer encoded value
        feature_name: name of the feature (e.g., 'customerType', 'assetCategory', 'transactionType', 'country')
        encodings: encodings dict from datasets metadata
    
    Returns:
        str: Original categorical value, or "Unknown" if the feature was missing/NaN, 
             or None if not found
    """
    # Check if it's a customer feature
    if 'customer_mappings' in encodings and feature_name in encodings['customer_mappings']:
        mapping = encodings['customer_mappings'][feature_name]
        if isinstance(encoded_value, (int, float)) and int(encoded_value) < len(mapping['categories']):
            return mapping['categories'][int(encoded_value)]
    
    # Check if it's an asset feature
    if 'asset_mappings' in encodings and feature_name in encodings['asset_mappings']:
        mapping = encodings['asset_mappings'][feature_name]
        if isinstance(encoded_value, (int, float)) and int(encoded_value) < len(mapping['categories']):
            return mapping['categories'][int(encoded_value)]
    
    # Check if it's a transaction feature
    if 'transaction_mappings' in encodings and feature_name in encodings['transaction_mappings']:
        mapping = encodings['transaction_mappings'][feature_name]
        if isinstance(encoded_value, (int, float)) and int(encoded_value) < len(mapping['categories']):
            return mapping['categories'][int(encoded_value)]
    
    # Check if it's a market feature
    if 'market_mappings' in encodings and feature_name in encodings['market_mappings']:
        mapping = encodings['market_mappings'][feature_name]
        if isinstance(encoded_value, (int, float)) and int(encoded_value) < len(mapping['categories']):
            return mapping['categories'][int(encoded_value)]
    
    return None

if __name__ == "__main__":
    preprocess_data()