# Project: Interpretable Models vs. Post-Hoc Explanations for Diabetic Readmission

This repository contains the end-to-end pipeline for predicting 30-day hospital readmission using the UCI Diabetes 130-US hospitals dataset. It evaluates the trade-offs between inherently interpretable models and black-box ensembles (XGBoost) paired with post-hoc explainability frameworks (SHAP, LIME, DiCE).

## Prerequisites
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (or Mamba)
- Git

## 1. Setup Environment
Clone the repository and create the environment using the provided YAML file:

```bash
# Clone the repository
git clone https://github.com/Codestronomer/diabetes-readmission-xai.git
cd diabetes_readmission_xai

# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate diabetes-readmission-xai
```

## 2. Data Ingestion
Before running the analysis, you must ingest the raw data. This script fetches the dataset from the UCI repository and performs an integrity check (Verification Gate 1).

```Bash
cd src
python data.py
```
Note: Ensure the output confirms "Status: PASSED" for dimensions (101766, 50).

## 3. Running the Analysis Pipeline

The project is structured into sequential phases managed through Jupyter Notebooks located in the `notebooks/` directory. Execute the notebooks in the following order.

### A. Data Preprocessing

Run `notebooks/02_preprocessing.ipynb`. This notebook performs the following tasks:

- Removes duplicate records, retaining only the first encounter for each patient.
- Groups ICD-9 diagnosis codes into broader diagnostic categories.
- Applies feature scaling and one-hot encoding to prepare the dataset for modeling.
- Addresses class imbalance by applying **SMOTE** to the training set (`X_train`) only.
- Saves the processed datasets and preprocessing artifacts to the `data/processed/` directory.
