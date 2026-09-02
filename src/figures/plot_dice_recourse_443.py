import os
import sys
import joblib
import numpy as np
import pandas as pd
import dice_ml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")


class XGBoostDiceWrapper:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.named_steps = pipeline.named_steps

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        else:
            X = pd.DataFrame(X, columns=self.named_steps['classifier'].get_booster().feature_names)
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        return self.pipeline.predict(X)


def run_dice_for_patient_443(data_dir=None, model_dir=None, output_dir=None):
    if data_dir is None:
        data_dir = os.path.join(ROOT_DIR, "data", "processed")
    if model_dir is None:
        model_dir = os.path.join(ROOT_DIR, "models")
    if output_dir is None:
        output_dir = os.path.join(ROOT_DIR, "results", "figures")

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()

    # Sanitize column names
    for df in [X_train, X_test]:
        df.columns = (df.columns
                      .str.replace('[', '_', regex=False)
                      .str.replace(']', '_', regex=False)
                      .str.replace('<', 'lt_', regex=False)
                      .str.replace('>', 'gt_', regex=False))

    # Load model
    raw_model = joblib.load(os.path.join(model_dir, "xgboost.pkl"))
    xgb_model = raw_model.named_steps['classifier'] if isinstance(raw_model, ImbPipeline) else raw_model

    # Find highest-risk readmitted patient
    probs = xgb_model.predict_proba(X_test)[:, 1]
    readmitted_mask = y_test == 1
    readmitted_indices = np.where(readmitted_mask)[0]
    readmitted_probs = probs[readmitted_mask]
    sorted_idx = readmitted_indices[np.argsort(-readmitted_probs)]
    patient_idx = sorted_idx[0]  # highest-risk readmitted patient
    print(f"Selected Patient #{patient_idx}: Actual=Readmitted, P(Readmit)={probs[patient_idx]:.4f}")

    patient_original = X_test.iloc[patient_idx].copy()
    patient_actual = y_test[patient_idx]
    patient_prob_original = xgb_model.predict_proba(patient_original.values.reshape(1, -1))[0, 1]
    print(f"Patient #{patient_idx}: Actual={'Readmitted' if patient_actual==1 else 'Not Readmitted'}, "
          f"P(Readmit)={patient_prob_original:.4f}")

    # Identify feature types
    continuous_features = [c for c in X_train.columns
                           if X_train[c].nunique() > 2
                           and not any(c.startswith(p) for p in ['age_', 'gender_', 'race_', 'diag_'])]

    # Immutable features
    immutable_prefixes = ('age_', 'gender_', 'race_', 'diag_1_', 'diag_2_', 'diag_3_')

    # Build DiCE data object
    train_df = X_train.copy()
    train_df['readmitted'] = y_train

    dice_data = dice_ml.Data(
        dataframe=train_df,
        continuous_features=continuous_features,
        outcome_name='readmitted'
    )

    # Wrap model
    wrapped = XGBoostDiceWrapper(raw_model)
    dice_model = dice_ml.Model(model=wrapped, backend="sklearn", model_type="classifier")

    # Create explainer
    dice_explainer = dice_ml.Dice(dice_data, dice_model, method="random")

    # Generate counterfactuals
    print(f"Generating counterfactuals for patient #{patient_idx}...")
    patient_df = patient_original.to_frame().T
    cf_result = dice_explainer.generate_counterfactuals(
        patient_df,
        total_CFs=3,
        desired_class=0,
        posthoc_sparsity_param=0.1,
        permitted_range=None
    )

    cf_example = cf_result.cf_examples_list[0]
    cf_df = cf_example.final_cfs_df
    if cf_df is None or len(cf_df) == 0:
        print("No counterfactuals generated")
        return
    # Drop the outcome column if present
    if 'readmitted' in cf_df.columns:
        cf_df = cf_df.drop(columns=['readmitted'])
    print(f"Generated {len(cf_df)} counterfactuals")

    # Compute probabilities for each counterfactual
    probs_original = xgb_model.predict_proba(patient_df)[0, 1]
    probs_cf = []
    for i in range(len(cf_df)):
        p = xgb_model.predict_proba(cf_df.iloc[[i]])[0, 1]
        probs_cf.append(p)

    # Find changed features for each CF
    changed_features = []
    for i in range(len(cf_df)):
        diff = cf_df.iloc[i] - patient_original
        changed = diff[diff != 0].index.tolist()
        # Filter to actionable features only
        changed = [f for f in changed if not any(f.startswith(p) for p in immutable_prefixes)]
        changed_features.append(changed)

    # === PLOT: Dual-pane figure ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Probability shift
    labels = ['Original'] + [f'CF {i+1}' for i in range(len(cf_df))]
    probs = [probs_original] + probs_cf
    colors_a = ['#d62728'] + ['#2ca02c'] * len(cf_df)

    bars_a = ax1.bar(labels, probs, color=colors_a, edgecolor='white', width=0.6)
    ax1.axhline(y=0.5, color='gray', lw=1, linestyle='--', label='Decision boundary')
    ax1.set_ylabel("Predicted P(Readmission)", fontsize=12)
    ax1.set_title("Panel A: Probability Shift via Counterfactuals", fontsize=13)
    ax1.set_ylim(0, max(probs) * 1.15)
    ax1.legend(fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    for bar, p in zip(bars_a, probs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{p:.3f}", ha='center', va='bottom', fontsize=11)

    # Panel B: Feature changes (for CF1 only, as representative)
    if changed_features[0]:
        cf1_diff = cf_df.iloc[0][changed_features[0]] - patient_original[changed_features[0]]
        feat_labels = changed_features[0][:10]  # Top 10 changed
        feat_vals = [cf1_diff[f] for f in feat_labels]

        colors_b = ['#d62728' if v > 0 else '#2ca02c' for v in feat_vals]
        ax2.barh(range(len(feat_labels)), feat_vals, color=colors_b, edgecolor='white', height=0.6)
        ax2.set_yticks(range(len(feat_labels)))
        ax2.set_yticklabels(feat_labels, fontsize=10)
        ax2.invert_yaxis()
        ax2.set_xlabel("Feature Change (Counterfactual − Original)", fontsize=12)
        ax2.set_title("Panel B: Actionable Feature Modifications (CF 1)", fontsize=13)
        ax2.axvline(x=0, color='gray', lw=0.8, linestyle='--')
    else:
        ax2.text(0.5, 0.5, "No feature changes", ha='center', va='center',
                 transform=ax2.transAxes, fontsize=14)
        ax2.set_title("Panel B: Actionable Feature Modifications (CF 1)", fontsize=13)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle(f"DiCE Algorithmic Recourse — Patient #{patient_idx}", fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "figure4_4_dice_recourse_443.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure 4.4 saved to: {out_path}")

    # Save counterfactuals CSV
    csv_path = os.path.join(ROOT_DIR, "results", "tables", f"patient_{patient_idx}_counterfactuals.csv")
    cf_df.to_csv(csv_path, index=False)
    print(f"Counterfactuals CSV saved to: {csv_path}")


if __name__ == "__main__":
    run_dice_for_patient_443()
