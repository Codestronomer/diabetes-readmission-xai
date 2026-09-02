import os
import sys
import joblib
import numpy as np
import pandas as pd
import lime.lime_tabular
import matplotlib.pyplot as plt

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")


def plot_lime_patient(data_dir=None, model_dir=None, output_dir=None,
                       patient_idx=15, suffix="4a"):
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
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
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

    X_train_sample = X_train.sample(min(5000, len(X_train)), random_state=42)

    # Create LIME explainer
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_sample.values,
        feature_names=X_test.columns.tolist(),
        class_names=['Not Readmitted', 'Readmitted'],
        mode='classification',
        kernel_width=0.75,
        random_state=42
    )

    # Generate explanation for specified patient
    exp = lime_explainer.explain_instance(
        data_row=X_test.iloc[patient_idx].values,
        predict_fn=xgb_model.predict_proba,
        num_features=10
    )

    prob = xgb_model.predict_proba(X_test.iloc[[patient_idx]])[:, 1][0]
    actual = y_test[patient_idx]

    # Get feature weights
    weights = dict(exp.local_exp[1])
    features = list(weights.keys())
    values = [weights[f] for f in features]
    labels = [X_test.columns[f] for f in features]

    # Color: green for negative (reduces risk), red for positive (increases risk)
    colors = ['#d62728' if v > 0 else '#2ca02c' for v in values]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(labels))
    ax.barh(y_pos, values, color=colors, edgecolor='white', height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Contribution Weight", fontsize=13)
    ax.set_title(
        f"LIME Local Explanation — Patient #{patient_idx} "
        f"(Actual: {'Readmitted' if actual == 1 else 'Not Readmitted'}, "
        f"Predicted P(Readmit) = {prob:.4f})",
        fontsize=12
    )
    ax.axvline(x=0, color='gray', lw=0.8, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"figure4_{suffix}_lime_patient{patient_idx}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Figure 4.{suffix} (Patient #{patient_idx}) saved to: {out_path}")


if __name__ == "__main__":
    plot_lime_patient(patient_idx=15, suffix="3a")
    plot_lime_patient(patient_idx=7, suffix="3b")
