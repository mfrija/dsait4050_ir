import os
import sys
import numpy as np
import pandas as pd

# Allow imports from src/ and from scripts/ (for preprocess.py)
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_scripts_dir)
sys.path.insert(0, _repo_root)
sys.path.insert(0, _scripts_dir)

from preprocess import load_datasets
from src.models.popularity import PopularityRecommender
from src.models.matrix_factorization import MFRecommender
from src.models.bpr_mf import BPRRecommender
from src.models.ncf import NCFRecommender
from src.models.content_based import ContentBasedRecommender
from src.evaluation.metrics import evaluate_model

K_VALUES = [10, 20]
MAX_K = max(K_VALUES)
NUM_ITEMS = 837

MODELS = [
    ("Popularity", PopularityRecommender()),
    ("MF_SVD_50", MFRecommender(n_components=50, n_iter=10, random_state=42)),
    ("BPR_MF", BPRRecommender(embedding_size=64, epochs=20, batch_size=2048, lr=1e-3)),
    ("NCF", NCFRecommender(embedding_size=64, epochs=10, batch_size=2048, lr=1e-3)),
    ("ContentBased", ContentBasedRecommender())
]


def run():
    processed_data_path = os.path.join(_repo_root, "data", "processed")
    results_path = os.path.join(_repo_root, "data", "results")
    os.makedirs(results_path, exist_ok=True)

    print("\n" + "=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)
    datasets = load_datasets(processed_data_path)
    variants = datasets["datasets"]
    print(f"Loaded {len(variants)} variants.\n")

    # ------------------------------------------------------------------
    # Load item (asset) features for Content-Based model
    # ------------------------------------------------------------------

    asset_path = os.path.join(processed_data_path, "asset_information.csv")
    asset_df = pd.read_csv(asset_path)

    # Use encoded categorical columns (already processed in preprocess.py)
    feature_cols = ["assetCategory", "assetSubCategory", "sector", "industry"]

    item_features = asset_df[feature_cols].values.astype(np.float32)

    print(f"Loaded item features: {item_features.shape}")

    all_rows = []  # one row per (model, variant)

    for model_name, model in MODELS:
        print("=" * 70)
        print(f"MODEL: {model_name}")
        print("=" * 70)

        for variant in variants:
            variant_id = variant["variant_id"]
            train_matrix = variant["train_rel_matrix"]
            test_matrix = variant["test_rel_matrix"]

            # Customer indices are the same for train and test (guaranteed by preprocess.py)
            customer_indices = np.arange(train_matrix.shape[0])

            # Re-fit fresh model per variant
            if model_name == "ContentBased":
                model.fit(train_matrix, item_features=item_features)
            else:
                model.fit(train_matrix)

            # Generate top-MAX_K recommendations for all customers
            recommendations = model.recommend(
                train_matrix, customer_indices, k=MAX_K, exclude_seen=True
            )

            metrics = evaluate_model(
                recommendations=recommendations,
                test_matrix=test_matrix,
                customer_indices=customer_indices,
                num_items=NUM_ITEMS,
                k_values=K_VALUES,
            )

            row = {"model": model_name, "variant_id": variant_id, **metrics}
            all_rows.append(row)

            print(f"  Variant {variant_id:2d}: NDCG@10={metrics.get('ndcg@10', 0):.4f}  NDCG@20={metrics.get('ndcg@20', 0):.4f}")

        print()

    # Save per-variant CSV
    df = pd.DataFrame(all_rows)
    per_variant_path = os.path.join(results_path, "metrics_per_variant.csv")
    df.to_csv(per_variant_path, index=False)

    # Aggregate: mean over variants
    metric_cols = [c for c in df.columns if c not in ("model", "variant_id")]
    summary = df.groupby("model")[metric_cols].mean().reset_index()
    summary_path = os.path.join(results_path, "metrics_summary.csv")
    summary.to_csv(summary_path, index=False)

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY (mean across variants)")
    print("=" * 70)
    for _, row in summary.iterrows():
        print(
            f"  {row['model']:<15}"
            f"  NDCG@10={row.get('ndcg@10', 0):.4f}"
            f"  NDCG@20={row.get('ndcg@20', 0):.4f}"
            f"  Coverage={row.get('catalog_coverage', 0):.3f}"
            f"  Gini={row.get('gini_index', 0):.3f}"
        )

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    run()
