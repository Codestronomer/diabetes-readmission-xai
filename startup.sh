#!/bin/bash

# Define the root directory name
PROJECT_DIR="diabetes_readmission_project"

# Create the root directory and navigate into it
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR" || exit

# Create the directory structure
echo "Creating directory structure..."
mkdir -p data/raw data/processed
mkdir -p notebooks
mkdir -p src/models src/explainability src/counterfactuals
mkdir -p experiments/run_template
mkdir -p models
mkdir -p results/tables results/figures
mkdir -p reports
mkdir -p references
mkdir -p logs/weekly logs/runs

# Create initial files
echo "Generating initial files..."
touch README.md
touch .gitignore
touch notebooks/01_eda.ipynb
touch notebooks/02_preprocessing.ipynb
touch src/__init__.py
touch src/data.py
touch src/features.py
touch src/evaluation.py
touch src/visualization.py
touch src/models/__init__.py
touch src/models/train_lr.py
touch src/models/train_dt.py
touch src/models/train_rf.py
touch src/models/train_xgb.py
touch src/explainability/__init__.py
touch src/explainability/shap_analysis.py
touch src/explainability/lime_analysis.py
touch src/counterfactuals/__init__.py
touch src/counterfactuals/dice_analysis.py
touch experiments/run_template/config.yaml
touch experiments/run_template/metrics.json
touch reports/final_report.pdf
touch references/bibliography.bib

# Populate the .gitignore with standard Python and ML exclusions
echo "Populating .gitignore..."
cat <<EOL > .gitignore
# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Conda
.conda

# Jupyter Notebook
.ipynb_checkpoints

# Python
__pycache__/
*.py[cod]
*$py.class
*.so

# Data and Models (Prevent pushing large files to remote)
data/raw/*
data/processed/*
models/*.pkl
models/*.joblib
models/*.h5
models/*.pt
experiments/**/*.pkl

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
EOL

# Initialize Git repository
echo "Initializing Git repository..."
git init
git add .
git commit -m "chore: initial phase 0 commit with project blueprint structure"

echo "Phase 0 Initialization Complete!"
echo "Next steps:"
echo "1. Run: cd $PROJECT_DIR"
echo "2. Save the provided YAML as environment.yml in this folder."
echo "3. Run: conda env create -f environment.yml"
echo "4. Run: conda activate diabetes-readmission-xai"
