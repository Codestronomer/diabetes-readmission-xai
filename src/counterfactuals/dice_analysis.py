import os
import joblib
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import dice_ml
import matplotlib.pyplot as plt

class XGBoostDiceWrapper:
    """Wrapper around XGBoost to ensure candidate DataFrames from DiCE 
    are strictly numeric before running predictions."""
    def __init__(self, model):
        self.model = model
        self.classes_ = model.classes_ if hasattr(model, 'classes_') else np.array([0, 1])

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.astype(np.float64)
        return self.model.predict_proba(X)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.astype(np.float64)
        return self.model.predict(X)

def run_dice_analysis(data_dir="../../data/processed", model_dir="../../models", output_dir="../../results"):
    print("--- DiCE Counterfactual Explanations ---")
    
    # Setup Output Directories
    fig_dir = os.path.join(output_dir, "figures")
    tbl_dir = os.path.join(output_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tbl_dir, exist_ok=True)
    
    # 1. Load XGBoost Model and Processed Data
    print("Loading XGBoost model and training/testing sets...")
    raw_xgb_model = joblib.load(os.path.join(model_dir, "xgboost.pkl"))
    
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

    # Force strict float64 numeric schema across all features
    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)

    # Recombine features and target for DiCE Data object
    train_df = X_train.copy()
    train_df['readmitted'] = y_train.astype(np.float64)
    
    # 2. Identify Continuous vs. Actionable Features
    continuous_features = [
        col for col in X_train.columns 
        if X_train[col].nunique() > 2 and not col.startswith(('age', 'gender', 'race', 'diag'))
    ]
    
    immutable_features = [
        col for col in X_train.columns 
        if col.startswith(('age', 'gender', 'race', 'diag_1', 'diag_2', 'diag_3'))
    ]
    actionable_features = [col for col in X_train.columns if col not in immutable_features]

    print(f"Total Features: {X_train.shape[1]}")
    print(f"Continuous Features Identified: {len(continuous_features)}")
    print(f"Actionable Features for Recourse: {len(actionable_features)}")

    # 3. Initialize DiCE Data and Wrapped Model
    d = dice_ml.Data(
        dataframe=train_df, 
        continuous_features=continuous_features, 
        outcome_name='readmitted'
    )
    
    # Wrap model to handle candidate DataFrame type conversions
    wrapped_model = XGBoostDiceWrapper(raw_xgb_model)
    m = dice_ml.Model(model=wrapped_model, backend="sklearn", model_type="classifier")
    
    exp = dice_ml.Dice(d, m, method="random")

    # 4. Select High-Risk True Positive Instance for Recourse Analysis
    probs = wrapped_model.predict_proba(X_test)[:, 1]
    high_risk_tp_indices = np.where((y_test == 1) & (probs > 0.65))[0]
    
    patient_idx = high_risk_tp_indices[0] if len(high_risk_tp_indices) > 0 else np.where(y_test == 1)[0][0]
    query_instance = X_test.iloc[[patient_idx]]
    orig_prob = probs[patient_idx]
    
    print(f"\nTarget Patient Index: {patient_idx}")
    print(f"Baseline Predicted Readmission Probability: {orig_prob:.4f}")

    # 5. Generate Counterfactual Explanations
    print("\nGenerating 3 Diverse Counterfactual Scenarios (Target Class: 0 - Not Readmitted)...")
    
    dice_exp = exp.generate_counterfactuals(
        query_instances=query_instance,
        total_CFs=3,
        desired_class=0,
        features_to_vary=actionable_features
    )

    # 6. Process and Export Results
    cf_df = dice_exp.cf_examples_list[0].final_cfs_df
    
    if cf_df is not None and not cf_df.empty:
        csv_path = os.path.join(tbl_dir, f"patient_{patient_idx}_counterfactuals.csv")
        cf_df.to_csv(csv_path, index=False)
        print(f"Counterfactual matrix saved to: {csv_path}")

        print("\n--- Key Recourse Changes (Original vs. Counterfactuals) ---")
        diff_cols = []
        for col in X_test.columns:
            orig_val = query_instance[col].values[0]
            cf_vals = cf_df[col].values
            if not np.all(np.isclose(cf_vals, orig_val)):
                diff_cols.append(col)
                print(f"Feature '{col}': Original = {orig_val} | CF Targets = {list(cf_vals)}")

        if diff_cols:
            plt.figure(figsize=(10, max(4, len(diff_cols) * 0.8)))
            plot_data = []
            
            orig_vals = query_instance[diff_cols].values[0]
            for i in range(len(cf_df)):
                cf_row = cf_df.iloc[i][diff_cols].values
                plot_data.append(cf_row)
                
            y_positions = np.arange(len(diff_cols))
            height = 0.2
            
            plt.barh(y_positions + height*1.5, orig_vals, height=height, label='Original Patient', color='#d9534f')
            for i, cf_row in enumerate(plot_data):
                plt.barh(y_positions - height*(i - 0.5), cf_row, height=height, label=f'CF Scenario {i+1}')

            plt.yticks(y_positions, diff_cols)
            plt.xlabel('Feature Value')
            plt.title(f'DiCE Counterfactual Levers for Patient {patient_idx} (Recourse to Class 0)', fontsize=12)
            plt.legend(loc='lower right')
            plt.tight_layout()
            
            fig_path = os.path.join(fig_dir, f"dice_patient_{patient_idx}_recourse.png")
            plt.savefig(fig_path, dpi=300)
            plt.close()
            print(f"Recourse visualization saved to: {fig_path}")
    else:
        print("No valid counterfactuals found within specified perturbation constraints.")

run_dice_analysis()