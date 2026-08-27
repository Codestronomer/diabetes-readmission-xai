import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, brier_score_loss,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")


def train_xgboost(data_dir=None, model_dir=None):
    if data_dir is None:
        data_dir = os.path.join(ROOT_DIR, "data", "processed")
    if model_dir is None:
        model_dir = os.path.join(ROOT_DIR, "models")

    print("--- Training XGBoost (SMOTE within CV folds) ---")

    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()

    # Sanitize feature names for XGBoost C++ backend
    for df in [X_train, X_test]:
        df.columns = (df.columns
                      .str.replace('[', '_', regex=False)
                      .str.replace(']', '_', regex=False)
                      .str.replace('<', 'lt_', regex=False)
                      .str.replace('>', 'gt_', regex=False))

    param_grid = {
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__n_estimators': [100, 300, 500],
        'classifier__subsample': [0.7, 0.8, 1.0],
        'classifier__colsample_bytree': [0.7, 0.8, 1.0],
    }

    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(
            tree_method='hist',
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_xgb = grid_search.best_estimator_
    print(f"Best Hyperparameters: {grid_search.best_params_}")

    y_test_probs = best_xgb.predict_proba(X_test)[:, 1]
    y_test_preds = best_xgb.predict(X_test)

    print("\nTest Metrics:")
    print(f"  AUC-ROC:        {roc_auc_score(y_test, y_test_probs):.4f}")
    print(f"  F1-Score:       {f1_score(y_test, y_test_preds):.4f}")
    print(f"  Accuracy:       {accuracy_score(y_test, y_test_preds):.4f}")
    print(f"  Precision:      {precision_score(y_test, y_test_preds):.4f}")
    print(f"  Recall (Sens):  {recall_score(y_test, y_test_preds):.4f}")
    print(f"  Brier Score:    {brier_score_loss(y_test, y_test_probs):.4f}")

    importances = best_xgb.named_steps['classifier'].feature_importances_
    features_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Gain_Importance': importances,
    }).sort_values(by='Gain_Importance', ascending=False)

    print("\nTop 5 Gain Importance Features:")
    print(features_df.head(5).to_string(index=False))

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "xgboost.pkl")
    joblib.dump(best_xgb, model_path)
    print(f"\nModel artifact serialized to: {model_path}\n")


if __name__ == "__main__":
    train_xgboost()
