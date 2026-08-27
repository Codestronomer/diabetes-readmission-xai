import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, brier_score_loss,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")


def train_random_forest(data_dir=None, model_dir=None):
    if data_dir is None:
        data_dir = os.path.join(ROOT_DIR, "data", "processed")
    if model_dir is None:
        model_dir = os.path.join(ROOT_DIR, "models")

    print("--- Training Random Forest (SMOTE within CV folds) ---")

    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 15, 20],
        'classifier__min_samples_split': [5, 10],
        'classifier__max_features': ['sqrt', 'log2'],
    }

    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(
            random_state=42, class_weight='balanced', n_jobs=-1,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    print(f"Best Hyperparameters: {grid_search.best_params_}")

    y_test_probs = best_rf.predict_proba(X_test)[:, 1]
    y_test_preds = best_rf.predict(X_test)

    print("\nTest Metrics:")
    print(f"  AUC-ROC:        {roc_auc_score(y_test, y_test_probs):.4f}")
    print(f"  F1-Score:       {f1_score(y_test, y_test_preds):.4f}")
    print(f"  Accuracy:       {accuracy_score(y_test, y_test_preds):.4f}")
    print(f"  Precision:      {precision_score(y_test, y_test_preds):.4f}")
    print(f"  Recall (Sens):  {recall_score(y_test, y_test_preds):.4f}")
    print(f"  Brier Score:    {brier_score_loss(y_test, y_test_probs):.4f}")

    importances = best_rf.named_steps['classifier'].feature_importances_
    features_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances,
    }).sort_values(by='Importance', ascending=False)

    print("\nTop 5 Gini Importance Features:")
    print(features_df.head(5).to_string(index=False))

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "random_forest.pkl")
    joblib.dump(best_rf, model_path)
    print(f"\nModel artifact serialized to: {model_path}\n")


if __name__ == "__main__":
    train_random_forest()
