# Interpretable Models vs. Post-Hoc Explanations for Diabetic Readmission

A comparative study for predicting 30-day hospital readmission of diabetic patients using the UCI Diabetes 130-US hospitals dataset. This project evaluates the trade-offs between inherently interpretable models (Logistic Regression, Decision Trees) and black-box ensembles (XGBoost, Random Forest) paired with post-hoc explainability frameworks (SHAP, LIME, DiCE).

## Project Structure

```
diabetes-readmission-xai/
├── data/
│   ├── raw/                          # Raw dataset (gitignored)
│   └── processed/                    # Preprocessed features and targets (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb        # Data preprocessing and feature engineering
│   └── Untitled.ipynb                # Scratch notebook
├── src/
│   ├── data.py                       # Data ingestion and validation
│   ├── features.py                   # Feature engineering utilities
│   ├── evaluation.py                 # Model evaluation metrics
│   ├── visualization.py              # Plotting utilities
│   ├── models/
│   │   ├── train_lr.py               # Logistic Regression training
│   │   ├── train_dt.py               # Decision Tree training
│   │   ├── train_rf.py               # Random Forest training
│   │   └── train_xgb.py              # XGBoost training
│   ├── explainability/
│   │   ├── shap_analysis.py          # SHAP explanation generation
│   │   ├── lime_analysis.py          # LIME explanation generation
│   │   └── xai_analysis.ipynb        # Interactive XAI analysis notebook
│   └── counterfactuals/
│       └── dice_analysis.py          # DiCE counterfactual generation
├── models/                           # Trained model artifacts (gitignored)
├── results/
│   ├── figures/                      # Generated plots and visualizations
│   └── tables/                       # CSV exports of analysis results
├── experiments/
│   └── run_template/                 # Experiment configuration templates
├── references/                       # Bibliography and citations
├── reports/                          # Final project reports
├── environment.yml                   # Conda environment specification
└── .gitignore                        # Git ignore rules
```

## Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (or [Mamba](https://mamba.readthedocs.io/))
- Git

## Setup

Clone the repository and create the conda environment:

```bash
# Clone the repository
git clone https://github.com/Codestronomer/diabetes-readmission-xai.git
cd diabetes-readmission-xai

# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate diabetes-readmission-xai
```

## Data Ingestion

Before running the analysis, ingest the raw data from the UCI repository:

```bash
cd src
python data.py
```

The script fetches the dataset and performs an integrity check (Verification Gate). Ensure the output confirms **"Status: PASSED"** for dimensions `(101766, 50)`.

## Analysis Pipeline

Execute the notebooks in the following order:

### 1. Exploratory Data Analysis

Run `notebooks/01_eda.ipynb` to explore the dataset, visualize distributions, and identify patterns.

### 2. Data Preprocessing

Run `notebooks/02_preprocessing.ipynb` which performs:

- Deduplication (retains first encounter per patient)
- ICD-9 diagnosis code grouping into 9 primary categories
- Feature scaling and one-hot encoding
- SMOTE oversampling for class imbalance (training set only)
- Saves processed data to `data/processed/`

### 3. Model Training

Train the four models using the scripts in `src/models/`:

| Model | Script | Type |
|-------|--------|------|
| Logistic Regression | `src/models/train_lr.py` | Interpretable |
| Decision Tree | `src/models/train_dt.py` | Interpretable |
| Random Forest | `src/models/train_rf.py` | Black-box |
| XGBoost | `src/models/train_xgb.py` | Black-box |

### 4. Explainability Analysis

#### SHAP (SHapley Additive exPlanations)

Global and local feature importance using `src/explainability/shap_analysis.py`.

#### LIME (Local Interpretable Model-agnostic Explanations)

Local interpretable explanations using `src/explainability/lime_analysis.py`.

#### DiCE (Diverse Counterfactual Explanations)

Counterfactual explanations for actionable recourse using `src/counterfactuals/dice_analysis.py`.

Run the interactive analysis notebook `src/explainability/xai_analysis.ipynb` for a comprehensive walkthrough.

## Results

Generated outputs are saved to:

- `results/figures/` — Visualization plots (SHAP summary, LIME explanations, DiCE counterfactuals)
- `results/tables/` — CSV exports of counterfactual analysis results

## Key Findings

This study compares:

1. **Predictive Performance** — AUC-ROC and F1 scores across models under class imbalance
2. **Interpretability** — Inherently interpretable models vs. post-hoc explanations
3. **Clinical Utility** — Fidelity and stability of counterfactual explanations with realistic medical constraints

## License

This project is for academic research purposes.
