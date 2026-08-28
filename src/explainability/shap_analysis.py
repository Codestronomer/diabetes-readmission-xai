import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt


def _patch_xgboost_shap_compat():
    """Monkey-patch SHAP to handle XGBoost 3.2.0 base_score serialization."""
    import shap.explainers._tree as shap_tree
    if hasattr(shap_tree, "_orig_decode_ubjson_buffer"):
        return  # already patched

    shap_tree._orig_decode_ubjson_buffer = shap_tree.decode_ubjson_buffer

    def _patched_decode_ubjson_buffer(fd):
        jmodel = shap_tree._orig_decode_ubjson_buffer(fd)
        try:
            learner_param = jmodel.get("learner", {}).get("learner_model_param", {})
            if "base_score" in learner_param:
                bs = str(learner_param["base_score"])
                if bs.startswith("[") and bs.endswith("]"):
                    learner_param["base_score"] = bs.strip("[]")
        except Exception:
            pass
        return jmodel

    shap_tree.decode_ubjson_buffer = _patched_decode_ubjson_buffer


def run_shap_analysis(data_dir="data/processed", model_dir="models", output_dir="results/figures"):
    print("--- SHAP Explainability Analysis ---")

    # Setup Directories
    os.makedirs(output_dir, exist_ok=True)

    # Load Model and Data
    print("Loading XGBoost model and test data...")
    raw_model = joblib.load(os.path.join(model_dir, "xgboost.pkl"))

    # Extract XGBClassifier from ImbPipeline if needed
    from imblearn.pipeline import Pipeline as ImbPipeline
    if isinstance(raw_model, ImbPipeline):
        xgb_model = raw_model.named_steps['classifier']
    else:
        xgb_model = raw_model

    # Load Test Data
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()

    # Sanitize feature column names
    X_test.columns = (X_test.columns
                      .str.replace('[', '_', regex=False)
                      .str.replace(']', '_', regex=False)
                      .str.replace('<', 'lt_', regex=False)
                      .str.replace('>', 'gt_', regex=False))

    # Sample subset for SHAP calculation speed
    X_test_sample = shap.sample(X_test, 1000, random_state=42)

    # Load 100 background samples for background distribution
    print("Loading 100 background samples from training set...")
    X_train_bg = shap.sample(
        pd.read_csv(os.path.join(data_dir, "X_train.csv")).rename(
            columns=lambda c: c.replace('[', '_').replace(']', '_').replace('<', 'lt_').replace('>', 'gt_')
        ),
        100, random_state=42
    )

    # Apply XGBoost 3.2.0 compatibility patch for TreeExplainer
    _patch_xgboost_shap_compat()

    print("\nGenerating SHAP Global Explanations (TreeExplainer)...")
    explainer = shap.TreeExplainer(xgb_model, data=X_train_bg)
    shap_values = explainer(X_test_sample)

    # Handle both 2D (single-output) and 3D (multi-output) SHAP values
    if len(shap_values.values.shape) == 3:
        shap_attrib = shap_values.values[:, :, 1]
    else:
        shap_attrib = shap_values.values

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_attrib, X_test_sample, show=False, max_display=20)
    plt.title("SHAP Summary Plot: Global Feature Impact on Readmission Risk", fontsize=14)
    plt.tight_layout()
    shap_fig_path = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(shap_fig_path, dpi=300)
    plt.close()
    print(f"SHAP summary plot saved to: {shap_fig_path}")


if __name__ == "__main__":
    run_shap_analysis()
