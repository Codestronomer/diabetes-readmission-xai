import os
import joblib
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

def run_lime_analysis(data_dir="../../data/processed", model_dir="../../models", output_dir="../../results/figures"):
    print("--- LIME Explainability Analysis ---")
    
    # 1. Setup Directories
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Load Model and Data
    print("Loading XGBoost model and test data...")
    xgb_model = joblib.load(os.path.join(model_dir, "xgboost.pkl"))
    
    # We use X_test for XAI to evaluate how the model explains unseen, real-world data
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    
    # Apply the same sanitization used during training
    X_test.columns = (X_test.columns
                      .str.replace('[', '_', regex=False)
                      .str.replace(']', '_', regex=False)
                      .str.replace('<', 'lt_', regex=False)
                      .str.replace('>', 'gt_', regex=False))

    # Sample a subset of the test data for SHAP to ensure reasonable computation time
    # SHAP TreeExplainer is fast, but rendering thousands of points can freeze plots
    X_test_sample = shap.sample(X_test, 1000, random_state=42)

    print("\nGenerating SHAP Global Explanations...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_sample)

    # Generate and save SHAP Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.title("SHAP Summary Plot: Global Feature Impact on Readmission Risk", fontsize=14)
    plt.tight_layout()
    shap_fig_path = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(shap_fig_path, dpi=300)
    plt.close()
    print(f"SHAP summary plot saved to: {shap_fig_path}")

    # ---------------------------------------------------------
    # PART B: LIME (Local Explainability)
    # ---------------------------------------------------------
    print("\nGenerating LIME Local Explanation for a high-risk patient...")
    
    # Find a patient in the test set who was ACTUALLY readmitted (y_test == 1)
    # and whom the model assigned a high probability of readmission.
    probs = xgb_model.predict_proba(X_test)[:, 1]
    true_positives = np.where((y_test == 1) & (probs > np.percentile(probs, 90)))[0]
    
    if len(true_positives) == 0:
        print("No high-confidence true positives found. Selecting a random true positive.")
        patient_idx = np.where(y_test == 1)[0][0]
    else:
        patient_idx = true_positives[0]
        
    print(f"Selected Patient Index: {patient_idx} | Actual: Readmitted | Predicted Prob: {probs[patient_idx]:.4f}")

    # Initialize LIME Tabular Explainer
    # LIME requires the training data distribution to build its local surrogate models
    # We load a small sample of X_train just to fit the LIME explainer
    X_train_sample = pd.read_csv(os.path.join(data_dir, "X_train.csv"), nrows=5000)
    X_train_sample.columns = X_test.columns # Apply sanitized columns
    
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_sample.values,
        feature_names=X_test.columns.tolist(),
        class_names=['Not Readmitted', 'Readmitted'],
        mode='classification',
        random_state=42
    )

    # Generate LIME explanation for the selected patient
    lime_exp = lime_explainer.explain_instance(
        data_row=X_test.iloc[patient_idx].values,
        predict_fn=xgb_model.predict_proba,
        num_features=10
    )

    # Save LIME plot
    lime_fig = lime_exp.as_pyplot_figure()
    lime_fig.set_size_inches(10, 6)
    plt.title(f"LIME Local Explanation (Patient {patient_idx})", fontsize=14)
    plt.tight_layout()
    lime_fig_path = os.path.join(output_dir, "lime_patient_explanation.png")
    plt.savefig(lime_fig_path, dpi=300)
    plt.close()
    print(f"LIME patient explanation saved to: {lime_fig_path}")

run_lime_analysis()