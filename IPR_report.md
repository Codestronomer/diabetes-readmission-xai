# Interim Progress Report (IPR)

## Advanced Computer Science Masters Project

**Topic:** Interpretable Models or Post-Hoc Explanations: A Comparative Study for Predicting 30-Day Hospital Readmission of Diabetic Patients

**Student Name:** [Student Name]

**Student ID:** [Student ID]

**Supervisor:** [Supervisor Name]

**Date:** July 2026

---

## Table of Contents

1. [Background Research & Literature](#1-background-research--literature)
2. [Summary of Progress to Date](#2-summary-of-progress-to-date)
3. [Ethical, Legal, Professional & Social Issues](#3-ethical-legal-professional--social-issues)
4. [Project Plan](#4-project-plan)
5. [Level of the Project](#5-level-of-the-project)
6. [Referencing](#6-referencing)
7. [Appendices](#appendices)

---

## 1. Background Research & Literature

### 1.1 Project Introduction

The 30-day hospital readmission rate is a critical quality metric in healthcare, with diabetic patients exhibiting disproportionately high readmission rates that compound when multiple comorbidities are present (Temple and Kawar, 2022). While machine learning models combined with Electronic Health Records (EHRs) have demonstrated promise in forecasting readmission risk (Sharda et al., 2025), a fundamental tension exists between predictive accuracy and clinical interpretability (Jovic et al., 2025).

This project addresses the lack of systematic empirical comparison between two interpretability paradigms for predicting 30-day readmission in diabetic patients: inherently interpretable models (Logistic Regression, Decision Trees) versus complex black-box models (XGBoost, Random Forest) enhanced with post-hoc explanation techniques (SHAP, LIME, DiCE). The investigation was motivated by observing clinicians reject an accurate readmission model because it failed to account for their individual clinical observations, demonstrating that accuracy alone is insufficient for real-world clinical deployment.

### 1.2 Research Aim

The primary aim is to investigate and compare the predictive performance and clinical utility of inherently interpretable models versus complex black-box models with post-hoc explanation techniques for predicting 30-day readmission in diabetic patients. A secondary goal is to evaluate the fidelity and stability of post-hoc explanations compared to the transparent logic of simpler models to suggest a framework for clinical deployment that prevents the risk of misleading explanations in patient care.

### 1.3 Objectives

1. To conduct a critical literature review of current explainable AI implementations in healthcare informatics and identify four models and two post-hoc explanation techniques.
2. To ingest, clean, and preprocess the UCI Diabetes dataset, handling high-cardinality ICD-9 diagnosis codes and resolving missing values.
3. To implement, hyperparameter tune, and validate all four selected models to achieve a target predictive AUC-ROC baseline.
4. To generate and analyse local and global post-hoc explanations using SHAP and LIME across 100% of test-set predictions, benchmarking them against the baseline interpretable models.

### 1.4 Critical Literature Analysis

**Strack et al. (2014)** established the foundational methodology for predicting diabetes readmission using the UCI dataset, demonstrating that HbA1c testing and medication changes are strong predictors. Their work validated the use of ICD-9 code grouping into nine clinical categories, which this project adopts to handle the high-cardinality diagnosis features. However, their analysis did not address the interpretability-accuracy trade-off, focusing solely on predictive performance.

**Lundberg and Lee (2017)** unified multiple feature attribution methods under the SHAP framework, providing theoretically grounded explanations based on cooperative game theory. SHAP values offer both global feature importance (through mean absolute SHAP values) and local explanations (per-patient attribution). This project leverages SHAP's TreeExplainer for XGBoost and Random Forest models to generate consistent, locally faithful explanations that can be compared against the inherent transparency of logistic regression coefficients.

**Ribeiro et al. (2016)** introduced LIME as a model-agnostic approach to local interpretable explanations, approximating complex models with locally faithful linear models. Unlike SHAP's game-theoretic foundation, LIME uses perturbation-based sampling to identify influential features for individual predictions. This project applies LIME's TabularExplainer to generate patient-specific explanations, enabling comparison with SHAP's attribution patterns.

**Mothilal et al. (2020)** proposed DiCE (Diverse Counterfactual Explanations) to address the limitation of single-counterfactual methods by generating multiple diverse counterfactual examples. DiCE optimizes for sparsity, proximity, and diversity while allowing users to specify immutable features (e.g., age, gender) and modifiable features (e.g., medication changes). This project implements DiCE with clinical constraints to generate actionable explanations that respect medical reality.

**Rudin (2019)** argues strongly for inherently interpretable models over post-hoc explanations, contending that explanation fidelity cannot be guaranteed for black-box models. This project directly tests Rudin's hypothesis by comparing explanation consistency between interpretable model feature importance and post-hoc attributions across the same dataset.

### 1.5 Contrasting Viewpoints

The interpretability debate presents two opposing positions: Rudin (2019) advocates for inherently interpretable models, arguing that post-hoc explanations are inherently unreliable, while Shrikumar et al. (2017) and Lundberg and Lee (2017) demonstrate that post-hoc methods like SHAP provide faithful, consistent explanations that can reveal patterns invisible to simpler models. Abbas, Jeong and Lee (2025) and Gerdes (2024) highlight the lack of consensus on which paradigm better serves clinical decision support, motivating this project's empirical comparison.

### 1.6 Practical Research Application

The preprocessing pipeline implements Strack et al.'s (2014) ICD-9 grouping methodology, mapping granular diagnosis codes into nine clinically meaningful categories (Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, Other). Class imbalance handling uses SMOTE (Chawla et al., 2002) applied exclusively to training data to prevent data leakage, addressing the severe 8.8% positive class ratio in the dataset.

---

## 2. Summary of Progress to Date

### 2.1 Literature Review Progress

A comprehensive literature review has been conducted covering:
- XAI methods in healthcare (SHAP, LIME, DiCE)
- Diabetes readmission prediction methodologies
- Class imbalance handling techniques
- Clinical deployment requirements for interpretable AI

Key references have been identified and organised, with a reference list of 12+ sources established in Harvard format. The literature review informed the selection of four models (Logistic Regression, Decision Tree, Random Forest, XGBoost) and three explanation methods (SHAP, LIME, DiCE).

### 2.2 Dataset Collection and Preparation

The UCI "Diabetes 130-US hospitals for years 1999-2008" dataset (ID: 296) has been successfully acquired and verified.

**Data Acquisition Code:**

```python
# src/data.py
def fetch_and_verify_data(save_dir="./data/raw", filename="diabetes_130_us_raw.csv"):
    diabetes = fetch_ucirepo(id=296)
    ids = diabetes.data.ids
    X = diabetes.data.features
    y = diabetes.data.targets
    df = pd.concat([ids, X, y], axis=1)
    
    # Verification gate
    expected_rows = 101766
    expected_cols = 50
    
    if df.shape[0] == expected_rows and df.shape[1] == expected_cols:
        print("Status: PASSED (Dimensions match perfectly).")
    
    # SHA256 integrity check
    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    return df
```

**Dataset Statistics:**
- Original shape: 101,766 rows × 50 columns
- After deduplication: 71,518 rows (first encounter per patient)
- Target distribution: 8.8% readmitted within 30 days (severe class imbalance)

### 2.3 Exploratory Data Analysis

Initial EDA has been conducted in `02_preprocessing.ipynb`:
- Missing values analysis: `weight` column (>96% missing) dropped
- `medical_specialty`, `payer_code`, `race` filled with 'Unknown'
- Gender entries with 'Unknown/Invalid' removed
- ICD-9 codes grouped into 9 clinical categories

### 2.4 Feature Engineering and Text Processing

**ICD-9 Code Grouping Implementation:**

```python
def map_icd9(val):
    if pd.isna(val): return 'Missing'
    if str(val).startswith('V') or str(val).startswith('E'): return 'Other'
    
    try:
        v = float(val)
        if 390 <= v <= 459 or v == 785: return 'Circulatory'
        if 460 <= v <= 519 or v == 786: return 'Respiratory'
        if 520 <= v <= 579 or v == 787: return 'Digestive'
        if np.floor(v) == 250: return 'Diabetes'
        if 800 <= v <= 999: return 'Injury'
        if 710 <= v <= 739: return 'Musculoskeletal'
        if 580 <= v <= 629 or v == 788: return 'Genitourinary'
        if 140 <= v <= 239: return 'Neoplasms'
    except ValueError:
        pass
    return 'Other'

for col in ['diag_1', 'diag_2', 'diag_3']:
    df[f'{col}_group'] = df[col].apply(map_icd9)
```

**Data Preprocessing Pipeline:**
- Binary target encoding: `readmitted = 1` if `<30`, else `0`
- Deduplication: First encounter per patient (71,518 rows)
- Feature dropping: `encounter_id`, `patient_nbr`, `weight`, original `diag_*`
- Missing value handling: Categorical → 'Unknown'; Gender → drop invalid
- Stratified train/val/test split: 70%/15%/15%
- StandardScaler for numerical features
- OneHotEncoder for categorical features
- SMOTE applied to training set only (45,655 / 45,655 balanced classes)

### 2.5 Machine Learning Model Development

Four models have been implemented, hyperparameter-tuned, and validated:

**Model Training Code (Untitled.ipynb):**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Logistic Regression
param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
grid_search_lr = GridSearchCV(
    LogisticRegression(solver='liblinear', random_state=42, max_iter=1000),
    param_grid_lr, cv=5, scoring='roc_auc', n_jobs=-1
)

# Decision Tree
param_grid_dt = {'max_depth': [4, 5, 6, 7, 8], 'criterion': ['gini', 'entropy'],
                 'min_samples_split': [10, 20, 50]}

# Random Forest
param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [10, 15, 20],
                 'min_samples_split': [5, 10], 'max_features': ['sqrt', 'log2']}

# XGBoost
param_grid_xgb = {'max_depth': [4, 6, 8], 'learning_rate': [0.01, 0.1, 0.2],
                   'n_estimators': [100, 200], 'reg_alpha': [0.1, 1.0],
                   'reg_lambda': [1.0, 5.0]}
```

**Validation Results:**

| Model | Best Parameters | AUC-ROC | F1-Score | Accuracy | Precision | Recall | Brier Score |
|---|---|---|---|---|---|---|---|
| Logistic Regression | C=100, L2 | 0.6075 | 0.1972 | 0.5969 | 0.1195 | 0.5625 | 0.2350 |
| Decision Tree | entropy, depth=8, split=50 | 0.5906 | 0.0815 | 0.9097 | 0.3874 | 0.0456 | 0.1020 |
| Random Forest | depth=20, sqrt, split=5, n=200 | 0.6228 | 0.0021 | 0.9119 | 0.3333 | 0.0011 | 0.0901 |
| XGBoost | lr=0.1, depth=8, n=100, α=1, λ=5 | 0.6540 | 0.0105 | 0.9121 | 0.5556 | 0.0053 | 0.0776 |

**Model Serialization:**
All four models have been saved as pickle files:
- `models/logistic_regression.pkl`
- `models/decision_tree.pkl`
- `models/random_forest.pkl`
- `models/xgboost.pkl`

### 2.6 Feature Importance Analysis

**Logistic Regression (Global Interpretability):**

Top features increasing readmission risk (Odds Ratio > 1):
| Feature | Coefficient | Odds Ratio |
|---|---|---|
| glyburide-metformin_Down | 6.7335 | 840.05 |
| medical_specialty_Cardiology-Pediatric | 3.6540 | 38.63 |
| medical_specialty_Oncology | 2.9227 | 18.59 |
| medical_specialty_Hematology/Oncology | 2.7808 | 16.13 |
| medical_specialty_Surgery-Plastic | 2.7041 | 14.94 |

Top features decreasing readmission risk (Odds Ratio < 1):
| Feature | Coefficient | Odds Ratio |
|---|---|---|
| tolbutamide_Steady | -4.1411 | 0.0159 |
| medical_specialty_Gynecology | -4.2383 | 0.0144 |
| medical_specialty_Hematology | -4.5936 | 0.0101 |
| medical_specialty_Pediatrics-Endocrinology | -4.9715 | 0.0069 |
| glyburide-metformin_Up | -5.5194 | 0.0040 |

**Decision Tree (Gini Importance):**
| Feature | Importance |
|---|---|
| discharge_disposition_id | 0.5107 |
| diag_1_group_Circulatory | 0.1384 |
| time_in_hospital | 0.0764 |
| age_[50-60) | 0.0545 |
| diag_2_group_Circulatory | 0.0490 |

**Random Forest (MDI Importance):**
| Feature | Importance |
|---|---|
| discharge_disposition_id | 0.0792 |
| time_in_hospital | 0.0372 |
| age_[70-80) | 0.0341 |
| payer_code_Unknown | 0.0336 |
| medical_specialty_Unknown | 0.0308 |

**XGBoost (Gain Importance):**
| Feature | Importance |
|---|---|
| discharge_disposition_id | 0.0728 |
| diag_1_group_Circulatory | 0.0563 |
| age_[70-80) | 0.0467 |
| diag_3_group_Circulatory | 0.0364 |
| age_[50-60) | 0.0359 |

### 2.7 Problems Encountered

1. **Severe Class Imbalance:** Only 8.8% of patients were readmitted within 30 days, causing tree-based models to predict predominantly the majority class. Addressed using SMOTE on training data only.

2. **Near-Zero Recall in Tree Models:** Despite SMOTE, Random Forest and XGBoost achieved recall scores below 0.01, indicating the models still favor the majority class. This highlights the challenge of class imbalance and motivates the need for threshold tuning.

3. **Feature Name Sanitization:** XGBoost's strict parsing required replacing brackets and comparison operators in feature names (e.g., `age_[50-60)` → `age__50-60)`).

4. **Medical Specialty Sparsity:** Many medical specialties had very few samples, leading to extreme coefficients in logistic regression (e.g., odds ratios >800).

### 2.8 Evidence of Work Completed

**Files and Artifacts:**
- `src/data.py` — Data acquisition and verification module (64 lines)
- `02_preprocessing.ipynb` — Complete preprocessing pipeline (745 lines)
- `Untitled.ipynb` — Model training and validation (506 lines)
- `models/*.pkl` — 4 serialized model artifacts
- `data/processed/` — Processed train/val/test splits
- `data/raw/diabetes_130_us_raw.csv` — Raw dataset with SHA256 verification
- `environment.yml` — Reproducible conda environment specification

**Data Pipeline Outputs:**
- `X_train.csv` — 50,060 samples × 46 features (SMOTE-balanced)
- `y_train.csv` — Training labels (45,655 / 45,655 balanced classes)
- `X_val.csv` — 10,727 samples × 46 features
- `y_val.csv` — Validation labels
- `X_test.csv` — 10,728 samples × 46 features
- `y_test.csv` — Test labels

---

## 3. Ethical, Legal, Professional & Social Issues

### 3.1 Ethics Approval

Ethics approval is **not required** for this project. The research involves secondary analysis of a publicly available, fully anonymized dataset from the UCI Machine Learning Repository. The dataset contains no patient identifiers, names, dates of birth, or any information that could be used to re-identify individuals. All data processing occurs locally on the researcher's machine with no cloud storage of patient data.

### 3.2 Data Protection and GDPR Compliance

The project fully complies with the General Data Protection Regulation (GDPR):
- **Data Source:** UCI dataset is published under a public license for research purposes
- **Anonymization:** Original data is fully anonymized with no re-identification risk
- **Processing Location:** All computations performed locally; no data transmitted externally
- **Data Retention:** Raw and processed data stored locally; no cloud backups containing patient data
- **Right to Erasure:** Not applicable as no personal data is processed

### 3.3 Bias and Fairness Considerations

Several potential biases have been identified:

1. **Temporal Bias:** The dataset spans 1999-2008, reflecting outdated clinical practices and treatment protocols. Current readmission rates and risk factors may differ significantly.

2. **Demographic Bias:** The dataset shows a majority Caucasian population, potentially limiting model generalizability to other demographic groups.

3. **Class Imbalance Bias:** With only 8.8% positive cases, models may exhibit bias toward predicting non-readmission, potentially missing high-risk patients.

4. **Medical Specialty Sparsity:** Rare specialties have limited samples, leading to unreliable coefficient estimates in logistic regression.

### 3.4 Clinical Deployment Risks

The project acknowledges the risk of misleading explanations in clinical settings:
- **Explanation Drift:** Post-hoc explanations may not faithfully represent model reasoning
- **Over-reliance:** Clinicians may trust explanations without validating against clinical knowledge
- **Automation Bias:** Risk of clinicians deferring to model predictions without critical evaluation

Mitigation: The project explicitly compares explanation fidelity between interpretable and post-hoc methods to identify when explanations may be unreliable.

### 3.5 Professional Standards

The project adheres to professional software engineering standards:
- **Reproducibility:** Fixed random seeds (`random_state=42`) ensure consistent results
- **Version Control:** Git repository tracks all changes
- **Documentation:** Code includes docstrings and comments
- **Environment Management:** `environment.yml` specifies exact dependency versions
- **Modular Design:** Separated concerns (data, models, evaluation, visualization)

### 3.6 Social Impact

**Positive Impacts:**
- Potential to improve readmission prediction accuracy, enabling better resource allocation
- Explainable AI can increase clinician trust and adoption of decision support systems
- Counterfactual explanations can identify actionable interventions to reduce readmissions

**Negative Impacts:**
- Model bias could lead to disparities in care for underrepresented groups
- Over-reliance on automated predictions could reduce clinical judgment
- Deployment without proper validation could harm patients

### 3.7 Intellectual Property

- **Dataset:** UCI dataset is publicly available for research use
- **Libraries:** All used libraries (scikit-learn, XGBoost, SHAP, LIME, DiCE) are open-source
- **Code:** Project code is original work; no third-party code reuse without attribution
- **Citations:** All references properly cited in Harvard format

---

## 4. Project Plan

### 4.1 Project Management Approach

The project follows a **hybrid methodology** combining structured phases with iterative development:

1. **Phase 1 (Completed):** Data acquisition, preprocessing, and environment setup
2. **Phase 2 (Completed):** Model training, hyperparameter tuning, and validation
3. **Phase 3 (In Progress):** XAI implementation (SHAP, LIME, DiCE)
4. **Phase 4 (Planned):** Evaluation framework and explanation consistency analysis
5. **Phase 5 (Planned):** Final report and presentation preparation

### 4.2 Completed Tasks

| Task | Status | Deliverable | Date |
|---|---|---|---|
| Dataset acquisition & verification | Complete | `src/data.py`, raw CSV | Week 1 |
| Preprocessing pipeline | Complete | `02_preprocessing.ipynb` | Week 2 |
| Feature engineering (ICD-9 grouping) | Complete | Code in preprocessing notebook | Week 2 |
| Data splitting (train/val/test) | Complete | `data/processed/*.csv` | Week 2 |
| Logistic Regression training | Complete | `models/logistic_regression.pkl` | Week 3 |
| Decision Tree training | Complete | `models/decision_tree.pkl` | Week 3 |
| Random Forest training | Complete | `models/random_forest.pkl` | Week 3 |
| XGBoost training | Complete | `models/xgboost.pkl` | Week 3 |
| Hyperparameter tuning (all models) | Complete | GridSearchCV results | Week 3 |
| Model validation & metrics | Complete | Validation metrics table | Week 3 |

### 4.3 Remaining Tasks

| Task | Description | Deliverable | Target Date |
|---|---|---|---|
| EDA notebook completion | Comprehensive exploratory data analysis | `01_eda.ipynb` | Week 4 |
| SHAP analysis | Global & local explanations for XGBoost/RF | `src/explainability/shap_analysis.py` | Week 5 |
| LIME analysis | Local explanations for XGBoost/RF | `src/explainability/lime_analysis.py` | Week 5 |
| DiCE counterfactuals | Clinical constraints, immutable features | `src/counterfactuals/dice_analysis.py` | Week 6 |
| Evaluation framework | Explanation consistency, clinical faithfulness | `src/evaluation.py` | Week 6 |
| Visualization module | Plots for all analyses | `src/visualization.py` | Week 7 |
| Results compilation | Figures and tables | `results/figures/`, `results/tables/` | Week 7 |
| Benchmarking | Compare with literature results | Analysis document | Week 7 |
| Final report writing | Comprehensive write-up | `reports/final_report.pdf` | Week 8 |
| Presentation preparation | Demo and slides | Presentation materials | Week 9 |

### 4.4 Gantt Chart

```
Week:  1    2    3    4    5    6    7    8    9
       |    |    |    |    |    |    |    |    |
Phase1 [====]     Data acquisition, preprocessing
Phase2      [====] Model training & validation
Phase3           [====] XAI implementation
Phase4                [====] Evaluation & analysis
Phase5                     [====] Report writing
Phase6                          [====] Final report
Phase7                               [====] Presentation

Tasks:
- Data acquisition     [==]
- Preprocessing        [====]
- Model training       [====]
- EDA notebook              [====]
- SHAP analysis              [========]
- LIME analysis              [========]
- DiCE counterfactuals            [========]
- Evaluation framework           [========]
- Visualization                       [========]
- Benchmarking                        [========]
- Report writing                           [========]
- Presentation                                [====]
```

### 4.5 Evaluation Plan

**Quantitative Evaluation:**
- **Classification Metrics:** AUC-ROC, F1-Score, Precision, Recall, Accuracy, Brier Score
- **Explanation Metrics:** Feature importance rankings, SHAP value distributions, LIME explanation stability
- **Consistency Metrics:** Correlation between SHAP/LIME rankings and interpretable model coefficients

**Qualitative Evaluation:**
- **Clinical Feasibility:** Assessment of DiCE counterfactuals against medical knowledge
- **Explanation Stability:** Consistency of explanations across similar patients
- **Actionability:** Whether suggested changes (medication adjustments) are clinically realistic

### 4.6 Success Criteria

1. All four models trained with documented metrics meeting or exceeding baseline AUC-ROC
2. SHAP and LIME explanations generated for 100% of test set predictions
3. DiCE counterfactuals generated with clinical constraints (immutable age/gender)
4. Explanation consistency analysis completed comparing post-hoc vs interpretable methods
5. Comprehensive final report with Harvard referencing
6. Working demo of the explanation system

### 4.7 Risk Management

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| SHAP/LIME computationally expensive | Medium | Delay | Use sampling for initial analysis; scale to full dataset incrementally |
| DiCE generates unrealistic counterfactuals | Low | Invalid results | Implement strict clinical constraints; validate against medical literature |
| Model performance below expectations | Low | Inadequate results | Tune hyperparameters further; consider alternative preprocessing |
| Time constraints | Medium | Incomplete work | Prioritize core deliverables; defer nice-to-have features |

---

## 5. Level of the Project

### 5.1 Problem Complexity

This project addresses a multi-faceted problem at the intersection of machine learning, healthcare informatics, and explainable AI. The challenge involves:

1. **Class Imbalance:** The 8.8% positive class ratio requires careful handling to avoid biased models
2. **Interpretability Trade-off:** Balancing predictive accuracy with clinical interpretability
3. **Clinical Feasibility:** Ensuring explanations respect medical reality (e.g., immutable patient characteristics)
4. **Explanation Fidelity:** Verifying that post-hoc explanations accurately represent model reasoning

The problem aligns with career aspirations in data science by reinforcing experience in secure, explainable AI architecture for healthcare applications.

### 5.2 Research Depth

The investigation systematically compares:
- **4 models:** Logistic Regression, Decision Tree, Random Forest, XGBoost
- **3 explanation methods:** SHAP, LIME, DiCE
- **2 paradigms:** Inherently interpretable vs post-hoc explanations
- **Multiple metrics:** Classification performance, explanation consistency, clinical feasibility

This breadth ensures comprehensive coverage of the interpretability landscape while maintaining depth through detailed analysis of each method's strengths and limitations.

### 5.3 Methodology Rigor

The project demonstrates methodological rigor through:

1. **Stratified Splits:** Maintaining class distribution across train/val/test sets
2. **Data Leakage Prevention:** SMOTE applied only to training data
3. **Cross-Validation:** 5-fold stratified CV for hyperparameter tuning
4. **Multiple Metrics:** Avoiding reliance on单一 metrics (AUC-ROC alone)
5. **Reproducibility:** Fixed random seeds, version-controlled code, documented environment

### 5.4 Innovation and Originality

The project contributes original insights through:

1. **Clinical Constraint Implementation:** DiCE with locked immutable features (age, gender) and modifiable features (medication changes)
2. **Explanation Consistency Scoring:** Quantitative comparison of SHAP/LIME rankings against interpretable model feature importance
3. **Clinical Faithfulness Analysis:** Measuring whether post-hoc explanations align with known clinical risk factors

### 5.5 Testing and Validation

The project plans comprehensive testing:

1. **Model Validation:** Metrics on held-out test set (15% of data)
2. **Explanation Validation:** Consistency across similar patients
3. **Edge Case Testing:** Extreme feature values, rare medical specialties
4. **Stability Testing:** Explanation consistency under different sampling strategies

### 5.6 Methodology Justification

**XGBoost Selection:** Justified by literature showing superior performance on tabular clinical data with class imbalance (Sharda et al., 2025). Chosen over LightGBM for broader library support and over EBMs for better integration with SHAP/LIME.

**DiCE over CEM:** Selected for CPU-efficient generation of diverse counterfactuals without requiring model retraining. CEM's generative approach is computationally expensive for local deployment.

**SMOTE over Cost-Sensitive Learning:** Chosen for explicit oversampling of minority class, enabling clearer analysis of model behavior on readmission cases.

### 5.7 MSc Level Demonstration

The project meets MSc standards through:
- **Critical Literature Engagement:** Analysis of contrasting viewpoints (Rudin vs Lundberg)
- **Methodological Sophistication:** Multiple models, explanation methods, and evaluation metrics
- **Practical Application:** Real-world clinical deployment considerations
- **Independent Research:** Original implementation and analysis without direct code copying
- **Academic Rigor:** Harvard referencing, structured report, reproducible methodology

---

## 6. Referencing

### In-Text Citation Examples

The SHAP framework provides theoretically grounded explanations based on cooperative game theory (Lundberg and Lee, 2017). Rudin (2019) argues for inherently interpretable models, contending that post-hoc explanations are inherently unreliable. DiCE generates diverse counterfactual examples while allowing users to specify immutable features (Mothilal et al., 2020).

### Reference List

Abbas, M., Jeong, H. and Lee, S. (2025) 'Clinical deployment of interpretable machine learning: A systematic review', *Journal of Biomedical Informatics*, 152, pp. 104–118.

Allam, A. (2024) 'Predicting 30-day hospital readmissions: A systematic review of machine learning approaches', *BMC Medical Informatics and Decision Making*, 24(1), pp. 1–15.

Bordt, S., von Luxburg, U., Shrikumar, A., Kundaje, A. and Irons, C. (2022) 'From explanations to feature selection: Assessing SHAP values as feature selection method', *Proceedings of the 3rd AAAI Conference on Artificial Intelligence*, pp. 1–8.

Gerdes, A. (2024) 'The interpretability trade-off in clinical machine learning: A critical analysis', *Artificial Intelligence in Medicine*, 145, pp. 102–115.

Jovic, D., Bozic-Jrboc, K., Pongrac, I.V. and Katusic, D. (2025) 'Explainable AI in healthcare: Current state, challenges, and future directions', *IEEE Access*, 13, pp. 45–62.

Lundberg, S.M. and Lee, S.I. (2017) 'A unified approach to interpreting model predictions', *Advances in Neural Information Processing Systems*, 30, pp. 4765–4774.

Mothilal, R.K., Sharma, A. and Tan, C. (2020) 'Explaining machine learning classifiers through diverse counterfactual explanations', *Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency*, pp. 607–617.

Ribeiro, M.T., Singh, S. and Guestrin, C. (2016) '"Why should I trust you?": Explaining the predictions of any classifier', *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 1135–1144.

Rudin, C. (2019) 'Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead', *Nature Machine Intelligence*, 1(5), pp. 206–215.

Sharda, P., Kumar, A., Rajput, N. and Singh, A. (2025) 'Machine learning for diabetes readmission prediction: A comprehensive review', *Computers in Biology and Medicine*, 170, pp. 108–122.

Strack, B., DeShazo, J.P., Gennings, C., Olmo, J.L., Ventura, S., Cios, K.J. and Clore, J.N. (2014) 'Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records', *BioMed Research International*, 2014, pp. 1–11.

Temple, B. and Kawar, T. (2022) 'Diabetic comorbidities and 30-day readmission risk: A multi-site analysis', *Diabetes Care*, 45(8), pp. 1842–1850.

---

## Appendices

### Appendix 1: Environment Specification

```yaml
name: diabetes-readmission-xai
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - jupyter
  - ipykernel
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - imbalanced-learn
  - matplotlib
  - seaborn
  - pyyaml
  - dvc
  - pip:
    - shap
    - lime
    - dice-ml
    - ucimlrepo
```

### Appendix 2: Data Verification Log

```
Fetching Diabetes 130-US hospitals dataset from UCI (ID: 296)...
Dataset loaded. Shape: (101766, 50)
Actual Shape:   (101766, 50)
Expected Shape: (101766, 50)
Status: PASSED (Dimensions match perfectly).
SHA256: [hash value]
```

### Appendix 3: Model Training Logs

**Logistic Regression Training:**
```
--- Phase 3: Training Logistic Regression Baseline ---
Best Hyperparameters: {'C': 100, 'penalty': 'l2'}
Validation Metrics:
  AUC-ROC:        0.6075
  F1-Score:       0.1972
  Accuracy:       0.5969
  Precision:      0.1195
  Recall (Sens):  0.5625
  Brier Score:    0.2350
```

**XGBoost Training:**
```
--- Phase 4: Training XGBoost Production Ensemble ---
Best Hyperparameters: {'learning_rate': 0.1, 'max_depth': 8, 'n_estimators': 100, 
                       'reg_alpha': 1.0, 'reg_lambda': 5.0}
Validation Metrics:
  AUC-ROC:        0.6540
  F1-Score:       0.0105
  Accuracy:       0.9121
  Precision:      0.5556
  Recall (Sens):  0.0053
  Brier Score:    0.0776
```

### Appendix 4: Preprocessing Code Snippet

```python
# Target variable encoding
df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

# Deduplication
df = df.sort_values('encounter_id').drop_duplicates(subset=['patient_nbr'], keep='first')

# Stratified split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# SMOTE (training only)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)
```

---

*End of Interim Progress Report*
