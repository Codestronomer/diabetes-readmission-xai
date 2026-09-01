import os
import joblib
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import shap
import lime
import lime.lime_tabular
from MLstatkit import Delong_test, Bootstrapping
import matplotlib.pyplot as plt


def _patch_xgboost_shap_compat():
    """Monkey-patch SHAP to handle XGBoost 3.2.0 base_score serialization."""
    import shap.explainers._tree as shap_tree
    if hasattr(shap_tree, "_orig_decode_ubjson_buffer"):
        return
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


# ======================================================================
# PART A: Clinical Faithfulness Analysis
# ======================================================================

def run_clinical_faithfulness(data_dir="data/processed", model_dir="models", output_dir="results/tables"):
    """
    Benchmark post-hoc XAI attributions (SHAP, LIME) against ground-truth
    model mechanics (LR log-odds, DT Gini impurity) via Spearman rank correlation.
    """
    print("=" * 60)
    print("Clinical Faithfulness Analysis")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load preprocessed data
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()

    # Sanitize feature names
    for df in [X_train, X_test]:
        df.columns = (df.columns
                      .str.replace('[', '_', regex=False)
                      .str.replace(']', '_', regex=False)
                      .str.replace('<', 'lt_', regex=False)
                      .str.replace('>', 'gt_', regex=False))

    feature_names = X_test.columns.tolist()

    # 2. Train fresh L2-regularized Logistic Regression
    print("\nTraining L2-regularized Logistic Regression for ground-truth benchmark...")
    lr_l2 = LogisticRegression(
        penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=42
    )
    lr_l2.fit(X_train.values, y_train)
    lr_importance = np.abs(lr_l2.coef_[0])

    # 3. Train fresh CART Decision Tree — Gini impurity as ground truth
    print("Training CART Decision Tree (Gini) for ground-truth benchmark...")
    dt = DecisionTreeClassifier(criterion='gini', random_state=42)
    dt.fit(X_train.values, y_train)
    dt_importance = dt.feature_importances_

    # 4. Sample 1000 test instances for XAI evaluation
    n_eval = min(1000, len(X_test))
    eval_indices = np.random.RandomState(42).choice(len(X_test), n_eval, replace=False)
    X_eval = X_test.iloc[eval_indices]

    # 5. SHAP explanations for LR
    print("Generating SHAP explanations for LR...")
    X_train_bg = shap.sample(X_train, 100, random_state=42)
    lr_explainer = shap.Explainer(lr_l2, X_train_bg)
    lr_shap_values = lr_explainer(X_eval)
    lr_shap_importance = np.abs(lr_shap_values.values).mean(axis=0)

    # 6. SHAP explanations for DT
    print("Generating SHAP explanations for DT...")
    dt_explainer = shap.Explainer(dt, X_train_bg)
    dt_shap_values = dt_explainer(X_eval, check_additivity=False)
    if len(dt_shap_values.shape) == 3:
        dt_shap_importance = np.abs(dt_shap_values.values[:, :, 1]).mean(axis=0)
    else:
        dt_shap_importance = np.abs(dt_shap_values.values).mean(axis=0)

    # 7. LIME explanations for LR
    print("Generating LIME explanations for LR...")
    lr_lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=['Not Readmitted', 'Readmitted'],
        mode='classification',
        kernel_width=0.75,
        random_state=42
    )

    lr_lime_avg = np.zeros(len(feature_names))
    n_lime_samples = min(200, n_eval)  # LIME is slower, use subset
    for i in range(n_lime_samples):
        exp = lr_lime_explainer.explain_instance(
            data_row=X_eval.iloc[i].values,
            predict_fn=lr_l2.predict_proba,
            num_features=len(feature_names)
        )
        for feat_name, weight in exp.local_exp[1]:
            lr_lime_avg[feat_name] += np.abs(weight)
    lr_lime_avg /= n_lime_samples

    # 8. LIME explanations for DT
    print("Generating LIME explanations for DT...")
    dt_lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=['Not Readmitted', 'Readmitted'],
        mode='classification',
        kernel_width=0.75,
        random_state=42
    )

    dt_lime_avg = np.zeros(len(feature_names))
    for i in range(n_lime_samples):
        exp = dt_lime_explainer.explain_instance(
            data_row=X_eval.iloc[i].values,
            predict_fn=dt.predict_proba,
            num_features=len(feature_names)
        )
        for feat_name, weight in exp.local_exp[1]:
            dt_lime_avg[feat_name] += np.abs(weight)
    dt_lime_avg /= n_lime_samples

    # 9. Compute Spearman rank correlations
    print("\n--- Spearman Rank Correlations ---")
    results = []

    pairs = [
        ("SHAP vs LR Coefficients", lr_shap_importance, lr_importance),
        ("LIME vs LR Coefficients", lr_lime_avg, lr_importance),
        ("SHAP vs DT Gini", dt_shap_importance, dt_importance),
        ("LIME vs DT Gini", dt_lime_avg, dt_importance),
    ]

    for label, xai_imp, gt_imp in pairs:
        rho, p_val = spearmanr(xai_imp, gt_imp)

        # Top-k overlap
        xai_top5 = set(np.argsort(xai_imp)[-5:])
        gt_top5 = set(np.argsort(gt_imp)[-5:])
        overlap5 = len(xai_top5 & gt_top5) / 5 * 100

        xai_top10 = set(np.argsort(xai_imp)[-10:])
        gt_top10 = set(np.argsort(gt_imp)[-10:])
        overlap10 = len(xai_top10 & gt_top10) / 10 * 100

        print(f"  {label}: rho={rho:.2f} (p={p_val:.2e}), Top-5 overlap={overlap5:.0f}%, Top-10 overlap={overlap10:.0f}%")
        results.append({
            'Benchmark': label,
            'Spearman_rho': round(rho, 2),
            'p_value': f'{p_val:.2e}',
            'Top5_Overlap_Pct': overlap5,
            'Top10_Overlap_Pct': overlap10
        })

    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "clinical_faithfulness.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    return results_df


# ======================================================================
# PART B: Statistical Validation & Hypothesis Testing
# ======================================================================

def run_statistical_validation(data_dir="data/processed", model_dir="models", output_dir="results/tables"):
    """
    DeLong tests (pairwise AUC comparison) and bootstrap CIs (1000 resamples).
    """
    print("\n" + "=" * 60)
    print("Statistical Validation & Hypothesis Testing")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load test data
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    X_test_sanitized = X_test.copy()
    X_test_sanitized.columns = (X_test_sanitized.columns
                      .str.replace('[', '_', regex=False)
                      .str.replace(']', '_', regex=False)
                      .str.replace('<', 'lt_', regex=False)
                      .str.replace('>', 'gt_', regex=False))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()

    # No global sanitization — each model may expect different column names.
    # We'll align per-model below.

    # 2. Load trained models and get predicted probabilities
    from imblearn.pipeline import Pipeline as ImbPipeline
    models = {}
    for name in ['logistic_regression', 'decision_tree', 'random_forest', 'xgboost']:
        path = os.path.join(model_dir, f"{name}.pkl")
        raw = joblib.load(path)
        if isinstance(raw, ImbPipeline):
            models[name] = raw.named_steps['classifier']
        else:
            models[name] = raw

    def get_expected_features(model):
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
        if hasattr(model, 'get_booster'):
            return model.get_booster().feature_names
        return None

    def align_features(X, expected_features):
        if expected_features is not None:
            return X[expected_features]
        return X

    probs = {}
    for name, model in models.items():
        expected = get_expected_features(model)
        if expected is not None:
            # Try original data first, then sanitized
            if all(f in X_test.columns for f in expected):
                X_aligned = X_test[expected]
            elif all(f in X_test_sanitized.columns for f in expected):
                X_aligned = X_test_sanitized[expected]
            else:
                raise ValueError(f"Model {name}: cannot find expected features in either dataset version")
        else:
            X_aligned = X_test
        probs[name] = model.predict_proba(X_aligned)[:, 1]

    model_display = {
        'xgboost': 'XGBoost',
        'random_forest': 'Random Forest',
        'decision_tree': 'Decision Tree',
        'logistic_regression': 'Logistic Regression'
    }

    # 3. Pairwise DeLong tests: XGBoost vs each
    print("\n--- DeLong Tests (XGBoost vs each model) ---")
    delong_results = []
    for name in ['random_forest', 'decision_tree', 'logistic_regression']:
        z, p_val, ci_xgb, ci_other, auc_xgb, auc_other, info = Delong_test(
            y_test, probs['xgboost'], probs[name],
            alpha=0.95, return_ci=True, return_auc=True, verbose=0
        )
        print(f"  XGBoost vs {model_display[name]}: Z={z:.2f}, p={p_val:.3e}, "
              f"AUC_XGB={auc_xgb:.4f}, AUC_{model_display[name]}={auc_other:.4f}")
        delong_results.append({
            'Comparison': f"XGBoost vs {model_display[name]}",
            'Z_score': round(z, 2),
            'p_value': f'{p_val:.3e}',
            'AUC_XGB': round(auc_xgb, 4),
            f'AUC_{model_display[name]}': round(auc_other, 4),
            'CI_XGB_lower': round(ci_xgb[0], 4),
            'CI_XGB_upper': round(ci_xgb[1], 4),
            'CI_other_lower': round(ci_other[0], 4),
            'CI_other_upper': round(ci_other[1], 4)
        })

    # 4. Bootstrap CIs (1000 resamples)
    print("\n--- Bootstrap CIs (1000 resamples) ---")
    bootstrap_results = []
    for name in ['xgboost', 'logistic_regression', 'decision_tree']:
        # ROC-AUC
        auc_score, auc_lower, auc_upper = Bootstrapping(
            y_test, probs[name], 'roc_auc', n_bootstraps=1000, confidence_level=0.95
        )
        print(f"  {model_display[name]} ROC-AUC: {auc_score:.4f} [{auc_lower:.4f}, {auc_upper:.4f}]")
        bootstrap_results.append({
            'Model': model_display[name],
            'Metric': 'ROC-AUC',
            'Empirical_Mean': round(auc_score, 4),
            'CI_Lower': round(auc_lower, 4),
            'CI_Upper': round(auc_upper, 4)
        })

        # F1 (only for XGBoost)
        if name == 'xgboost':
            f1_score_val, f1_lower, f1_upper = Bootstrapping(
                y_test, probs[name], 'f1', n_bootstraps=1000, confidence_level=0.95, threshold=0.5
            )
            print(f"  {model_display[name]} F1: {f1_score_val:.4f} [{f1_lower:.4f}, {f1_upper:.4f}]")
            bootstrap_results.append({
                'Model': model_display[name],
                'Metric': 'F1-Score',
                'Empirical_Mean': round(f1_score_val, 4),
                'CI_Lower': round(f1_lower, 4),
                'CI_Upper': round(f1_upper, 4)
            })

    # Save all results
    delong_df = pd.DataFrame(delong_results)
    bootstrap_df = pd.DataFrame(bootstrap_results)

    combined_path = os.path.join(output_dir, "statistical_validation.csv")
    with open(combined_path, 'w') as f:
        f.write("DeLong Test Results\n")
        delong_df.to_csv(f, index=False)
        f.write("\nBootstrap CI Results\n")
        bootstrap_df.to_csv(f, index=False)
    print(f"\nResults saved to: {combined_path}")

    return delong_df, bootstrap_df


# ======================================================================
# PART C: Explanation Stability & Cross-Method Alignment
# ======================================================================

def run_explanation_stability(data_dir="data/processed", model_dir="models", output_dir="results/tables"):
    """
    Measure cross-method agreement between TreeSHAP and LIME across
    identical predictions via Spearman rank correlation.
    """
    print("\n" + "=" * 60)
    print("Explanation Stability & Cross-Method Alignment")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Load XGBoost model
    raw_model = joblib.load(os.path.join(model_dir, "xgboost.pkl"))
    from imblearn.pipeline import Pipeline as ImbPipeline
    if isinstance(raw_model, ImbPipeline):
        xgb_model = raw_model.named_steps['classifier']
    else:
        xgb_model = raw_model

    # Load and sanitize data
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    X_test.columns = (X_test.columns
                      .str.replace('[', '_', regex=False)
                      .str.replace(']', '_', regex=False)
                      .str.replace('<', 'lt_', regex=False)
                      .str.replace('>', 'gt_', regex=False))
    feature_names = X_test.columns.tolist()

    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_train.columns = (X_train.columns
                      .str.replace('[', '_', regex=False)
                      .str.replace(']', '_', regex=False)
                      .str.replace('<', 'lt_', regex=False)
                      .str.replace('>', 'gt_', regex=False))

    # Sample 1000 random test instances
    n_eval = min(1000, len(X_test))
    rng = np.random.RandomState(42)
    eval_indices = rng.choice(len(X_test), n_eval, replace=False)
    X_eval = X_test.iloc[eval_indices]

    # Apply XGBoost 3.2.0 compatibility patch
    _patch_xgboost_shap_compat()

    # TreeSHAP explanations
    print("Generating TreeSHAP explanations for 1000 patients...")
    X_train_bg = shap.sample(X_train, 100, random_state=42)
    shap_explainer = shap.TreeExplainer(xgb_model, data=X_train_bg)
    shap_values = shap_explainer(X_eval)
    # Use class 1 (readmitted) SHAP values
    if len(shap_values.values.shape) == 3:
        shap_attrib = shap_values.values[:, :, 1]
    else:
        shap_attrib = shap_values.values

    # LIME explanations
    print("Generating LIME explanations for 1000 patients (this may take a few minutes)...")
    X_train_sample = X_train.sample(min(5000, len(X_train)), random_state=42)
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_sample.values,
        feature_names=feature_names,
        class_names=['Not Readmitted', 'Readmitted'],
        mode='classification',
        kernel_width=0.75,
        random_state=42
    )

    lime_attrib = np.zeros((n_eval, len(feature_names)))
    for i in range(n_eval):
        exp = lime_explainer.explain_instance(
            data_row=X_eval.iloc[i].values,
            predict_fn=xgb_model.predict_proba,
            num_features=len(feature_names)
        )
        for feat_idx, weight in exp.local_exp[1]:
            lime_attrib[i, feat_idx] = weight
        if (i + 1) % 200 == 0:
            print(f"  LIME progress: {i+1}/{n_eval}")

    # Compute per-patient Spearman rank correlations
    print("\nComputing cross-method rank correlations...")
    patient_rhos = []
    for i in range(n_eval):
        shap_ranks = pd.Series(np.abs(shap_attrib[i])).rank(ascending=False)
        lime_ranks = pd.Series(np.abs(lime_attrib[i])).rank(ascending=False)
        rho, _ = spearmanr(shap_ranks, lime_ranks)
        patient_rhos.append(rho)

    mean_rho = np.mean(patient_rhos)
    print(f"  Mean Spearman Rank Correlation (SHAP, LIME): {mean_rho:.2f}")

    # Top-k feature alignment across all patients
    shap_global = np.abs(shap_attrib).mean(axis=0)
    lime_global = np.abs(lime_attrib).mean(axis=0)

    shap_top3 = set(np.argsort(shap_global)[-3:])
    lime_top3 = set(np.argsort(lime_global)[-3:])
    top3_overlap = len(shap_top3 & lime_top3) / 3 * 100

    shap_top5 = set(np.argsort(shap_global)[-5:])
    lime_top5 = set(np.argsort(lime_global)[-5:])
    top5_overlap = len(shap_top5 & lime_top5) / 5 * 100

    print(f"  Top 3 Feature Alignment Rate: {top3_overlap:.1f}%")
    print(f"  Top 5 Feature Alignment Rate: {top5_overlap:.1f}%")

    # Save results
    results = pd.DataFrame({
        'Metric': [
            'Mean Spearman Rank Correlation (SHAP, LIME)',
            'Top 3 Feature Alignment Rate (%)',
            'Top 5 Feature Alignment Rate (%)',
            'N_patients_evaluated',
        ],
        'Value': [round(mean_rho, 2), round(top3_overlap, 1), round(top5_overlap, 1), n_eval]
    })
    csv_path = os.path.join(output_dir, "explanation_stability.csv")
    results.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    return results


if __name__ == "__main__":
    run_clinical_faithfulness()
    run_statistical_validation()
    run_explanation_stability()
