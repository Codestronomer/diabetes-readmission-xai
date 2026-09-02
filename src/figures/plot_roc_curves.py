import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")


def plot_roc_curves(data_dir=None, model_dir=None, output_dir=None):
    if data_dir is None:
        data_dir = os.path.join(ROOT_DIR, "data", "processed")
    if model_dir is None:
        model_dir = os.path.join(ROOT_DIR, "models")
    if output_dir is None:
        output_dir = os.path.join(ROOT_DIR, "results", "figures")

    os.makedirs(output_dir, exist_ok=True)

    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()

    # Sanitize for XGBoost
    X_test_xgb = X_test.copy()
    X_test_xgb.columns = (X_test_xgb.columns
                          .str.replace('[', '_', regex=False)
                          .str.replace(']', '_', regex=False)
                          .str.replace('<', 'lt_', regex=False)
                          .str.replace('>', 'gt_', regex=False))

    models = {
        "Logistic Regression": ("logistic_regression.pkl", X_test),
        "Decision Tree": ("decision_tree.pkl", X_test),
        "Random Forest": ("random_forest.pkl", X_test),
        "XGBoost": ("xgboost.pkl", X_test_xgb),
    }

    colors = {
        "Logistic Regression": "#1f77b4",
        "Decision Tree": "#ff7f0e",
        "Random Forest": "#2ca02c",
        "XGBoost": "#d62728",
    }

    plt.figure(figsize=(10, 8))

    for name, (filename, X_input) in models.items():
        raw = joblib.load(os.path.join(model_dir, filename))
        if isinstance(raw, ImbPipeline):
            model = raw.named_steps['classifier']
        else:
            model = raw

        probs = model.predict_proba(X_input)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)

        plt.plot(fpr, tpr, color=colors[name], lw=2,
                 label=f"{name} (AUC = {auc:.4f})")

    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Chance")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate", fontsize=13)
    plt.ylabel("True Positive Rate", fontsize=13)
    plt.title("ROC Performance Progression Across Candidate Models", fontsize=14)
    plt.legend(loc="lower right", fontsize=11, frameon=True)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "figure4_1_roc_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Figure 4.1 saved to: {out_path}")


if __name__ == "__main__":
    plot_roc_curves()
