import os
import sys
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "explainability"))
from shap_analysis import _patch_xgboost_shap_compat

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")


def plot_shap_top10(data_dir=None, model_dir=None, output_dir=None):
    if data_dir is None:
        data_dir = os.path.join(ROOT_DIR, "data", "processed")
    if model_dir is None:
        model_dir = os.path.join(ROOT_DIR, "models")
    if output_dir is None:
        output_dir = os.path.join(ROOT_DIR, "results", "figures")

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    raw = joblib.load(os.path.join(model_dir, "xgboost.pkl"))
    from imblearn.pipeline import Pipeline as ImbPipeline
    if isinstance(raw, ImbPipeline):
        xgb_model = raw.named_steps['classifier']
    else:
        xgb_model = raw

    # Load and sanitize data
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    X_test.columns = (X_test.columns
                      .str.replace('[', '_', regex=False)
                      .str.replace(']', '_', regex=False)
                      .str.replace('<', 'lt_', regex=False)
                      .str.replace('>', 'gt_', regex=False))

    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_train.columns = (X_train.columns
                       .str.replace('[', '_', regex=False)
                       .str.replace(']', '_', regex=False)
                       .str.replace('<', 'lt_', regex=False)
                       .str.replace('>', 'gt_', regex=False))

    X_test_sample = shap.sample(X_test, 1000, random_state=42)
    X_train_bg = shap.sample(X_train, 100, random_state=42)

    # Compute SHAP values
    _patch_xgboost_shap_compat()
    explainer = shap.TreeExplainer(xgb_model, data=X_train_bg)
    shap_values = explainer(X_test_sample)

    if len(shap_values.values.shape) == 3:
        shap_attrib = shap_values.values[:, :, 1]
    else:
        shap_attrib = shap_values.values

    # Compute mean |SHAP| per feature
    mean_abs_shap = np.abs(shap_attrib).mean(axis=0)
    feature_names = X_test_sample.columns.tolist()

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean|SHAP|': mean_abs_shap
    }).sort_values('Mean|SHAP|', ascending=False).head(10)

    # Plot horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(importance_df)), importance_df['Mean|SHAP|'].values,
                   color='#d62728', edgecolor='white', height=0.7)
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['Feature'].values, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value|", fontsize=13)
    ax.set_title("Top-10 Global Feature Attributions via TreeSHAP", fontsize=14)

    for i, (val, name) in enumerate(zip(importance_df['Mean|SHAP|'].values,
                                         importance_df['Feature'].values)):
        ax.text(val + 0.001, i, f"{val:.4f}", va='center', fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "figure4_2_shap_top10.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Figure 4.2 saved to: {out_path}")


if __name__ == "__main__":
    plot_shap_top10()
