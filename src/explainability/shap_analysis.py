import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

def run_shap_analysis(data_dir="../../data/processed", model_dir="../../models", output_dir="../../results/figures"):
    print("--- SHAP Explainability Analysis ---")
    
    # Setup Directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Model and Data
    print("Loading XGBoost model and test data...")
    xgb_model = joblib.load(os.path.join(model_dir, "xgboost.pkl"))
    
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

    print("\nGenerating SHAP Global Explanations...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_sample)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.title("SHAP Summary Plot: Global Feature Impact on Readmission Risk", fontsize=14)
    plt.tight_layout()
    shap_fig_path = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(shap_fig_path, dpi=300)
    plt.close()
    print(f"SHAP summary plot saved to: {shap_fig_path}")
    
run_shap_analysis()