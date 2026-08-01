**TOPIC: Interpretable Models or Post-Hoc Explanations/ A Comparative Study for Predicting 30-Day Hospital Readmission of Diabetic Patients**

**MODULE TITLE: ADVANCED COMPUTER SCIENCE MASTERS PROJECT** 

**NAME:**

**STUDENT ID:**

**1.0	Introduction**

The 30-day readmission rate is a major outcome measurement in hospital quality assessment and a major contributor to unnecessary healthcare spending that could be avoided (Allam, 2024). Diabetic patients have a higher rate of readmission which is a risk that compounds dramatically when multiple comorbidities are involved in their clinical care (Temple and Kawar, 2022). While Electronic Health Records (EHRs) combined with machine learning have shown immense promise in forecasting this risk (Sharda et al., 2025), there is a fundamental conflict between predictions and interpretability in clinical machine learning systems (Jovic et al., 2025). Recently, post-hoc explanation methods such as Diverse Counterfactual Explanations (DiCE) have been proposed to explain “black box” predictions (Bordt et al., 2022). However, there is no agreement on the advantages of inherently interpretable models against these post-hoc explanations for clinical decision support (CDS) (Abbas, Jeong and Lee, 2025; Gerdes, 2024).

The problem inspiring this project is the lack of a systematic, empirical comparison between these two interpretability paradigms for the problem of predicting 30-day readmission in diabetic patients, specifically in terms of predictive performance under class imbalance and their clinical feasibility in generating counterfactuals. This project was inspired by a personal observation where I watched clinicians actively reject an accurate readmission model because it failed to account for their individual observation of a patient who they saw as having a high risk of readmission. This experience has shown that accuracy is not enough for deployment in the real world for clinical use. To address this gap, this investigative project systematically compares an inherently interpretable model against an advanced ensemble pipeline using XGBoost paired with DiCE. The focus of the investigation is to identify the paradigm that maintains predictive stability when severe class imbalance occurs, and produces counterfactual explanations that respect realistic medical constraints (e.g., locking immutable features like age, while allowing actionable changes like medication adjustments). The rest of this proposal includes project aim, core requirements, secondary research aims and the methodology to be used.

**2.0	Project Aim**  
The primary aim of this research project is to investigate and compare the predictive performance and the clinical utility of inherently interpretable models versus complex black box models with post-hoc explanation techniques for predicting the 30-day readmission for diabetic patients. A secondary goal is also to evaluate the “fidelity” and “stability” of post-hoc explanations compared to the clear logic of simple models in order to suggest a framework for clinical deployment, preventing the risk of “misleading explanations” in patient care.

**Smart Objectives**

1. To conduct a critical literature review of current explainable AI implementations in healthcare informatics and identify four models and two post-hoc explanation techniques.

2. To ingest, clean, and preprocess the UCI Diabetes dataset, handling high-cardinality ICD-9 diagnosis code and resolving missing values.

3. To implement, hyperparameter tune, and validate all four selected models to achieve a target predictive AUC-ROC baseline

4. To generate and analyse local and global Post-hoc explanations using SHAP and LIME across 100% of test-set predictions, benchmarking them against the baseline interpretable models.

**2.1	Core Project requirements**

1. **Model Implementation:** To design and implement two inherently interpretable models (Logistic Regression, Decision Trees) and also two complex models (XGBoost, Random Forest)  
2. **Explanation Integration:** To apply Shapley Additive Explanations (SHAP) and LIME to the complex models to generate local and global feature importance.  
3. **Data Pipeline:** To develop a reliable and robust preprocessing pipeline for the “Diabetes 130-US hospitals for years 1990-2008” dataset, that can handle missing values and categorical encoding.  
4. **Evaluation Metrics:** To establish a baseline using AUC-ROC and F1-Score, while also measuring the “Explanation Consistency” across the various post-hoc methods

**2.2	Advanced Project Aims**

1. **Clinical Faithfulness Analysis:** To quantitatively compare the attributions produced by SHAP with features rankings of the interpretable models in order to identify “explanation drift”.  
2. **Benchmarking:** To compare and contrast the project’s predictive accuracy and interpretability metrics against state-of-the-art results found in recent health informatics literature.

**3.0	Secondary Research Aims (Literature Review)**  
The primary challenges in predicting 30-day hospital readmissions for diabetic patients involve navigating severe data class imbalance and meeting the clinical requirement for actionable explanations. The literature review is needed to determine which predictive models are best suited to the UCI Diabetes dataset, and which counterfactual methods are suitable for local deployment. The review will be conducted to validate comparison of XGBoost with inherently interpretable alternatives like EBMs. It will also evaluate counterfactual approaches such as DiCE and CEM and support the choice of DiCE when using CPU. The desired outcome is to set up empirical benchmarks for AUC and F1, and to identify if there is novelty in imposing clinical feasibility constraints. The sub-objectives of the secondary research are:

* To select and justify the algorithm stack (XGBoost) by reviewing literatures of other classifiers (LightGBM and EBMs) for tabular clinical data with high class imbalance. Also, conducting an assessment of state-of-the-art generative frameworks for local CPU constraints and custom optimisation goals to justify adopting Diverse Counterfactual Explanations (DiCE) over alternative methods like CEM.  
* To determine critical clinical factors by analyzing historic treatments within the UCI Diabetes dataset to pinpoint the most predictive clinical variables (e.g., HbA1c testing rates, specific medication changes). These identified variables will be cross-referenced with established clinical guidelines to categorize patient features into immutable constraints (e.g., locking age) and modifiable features (e.g., altering dosages), ensuring that the generated counterfactual uncertainties remain under strict clinical plausibility.  
* To gather state-of-the-art results from existing peer-reviewed studies using the same UCI dataset as an absolute point of reference. Specifically, to create scores to measure predictive accuracy (AUC, F1 score) and explanation quality (sparsity, proximity). These metrics will be used to determine if applying clinical feasibility constraints in XGBoost plus DiCE lowers or maintains state of the art performance.

**4.0	Primary Research**  
Utilising the UCI diabetes dataset, the project will proceed in these stages:

1. **Data Processing:** Data cleaning, feature engineering (e.g grouping diagnosis codes), and addressing class imbalance using SMOTE.  
2. **Artifact Development:** Utilising python and frameworks such as Scikit-Learn, XGBoost, and the SHAP/LIME libraries to build and train models.  
3. **Training & Testing:** Using K-fold cross-validation to ensure model stability and generalisability.  
4. **Evaluation:** Comparing the "Global Interpretability" (overall feature importance) and "Local Interpretability" (patient-specific predictions) across all models.

**5.0	Project Resources**  
**(i) Hardware:** A Macbook Pro (M-series) will be used for local development and an AWS EC2 instance if additional compute is required for hyperparameter tuning. (ii) **Software:** Python 3.9+, Jupyter Notebook, pandas, Scikit-learn, XGBoost, and lastly SHAP and LIME libraries. (iii) **Temporal resources:** Due to time limitations, this investigation is expected to be completed within 9 weeks.

**6.0	Project risks and their mitigation**

| No | Risk Description | Probability | Possible Effects | Mitigation Methods |
| :---- | :---- | :---- | :---- | :---- |
| **1** | High Data Imbalance | High | Model ignores readmitted cases | Use SMOTE or cost-sensitive learning. |
| **2** | Inconsistent Explanations | Medium | SHAP and LIME provide conflicting results | Use "Stability" metrics to evaluate explanations. |
| **3** | Project Scope Creep | Low | Incomplete analysis | Stick to 4 core models and 2 XAI methods. |

**7.1	Project Outcome**

The project’s expected outcome is a reproducible, empirical comparative framework designed to assess the operation safety of deploying “black-box” models in a clinical environment. The core deliverable will serve as a multi-metric scorecard that cross-examines classification performance (AUC-ROC and F1-Score) against explanation fidelity.

**7.2	Lessons to be learned**

This project directly supports my career objectives in Data science by reinforcing my experience and expertise in secure, explainable architecture.

Ultimately, I aim to master the mathematical fundamentals of SHAP values. This investigation will elevate my skills from simply applying 'plug-and-play' machine learning libraries to achieving a profound, foundational understanding of complex XAI algorithms.

**Reference list**

Cai, Z., Takabi, D., Guo, S. and Zou, Y. (eds) (2025) *Wireless Artificial Intelligent Computing Systems and Applications*, *Lecture Notes in Computer Science*. Springer Nature Switzerland. Available at: https://doi.org/10.1007/978-3-031-71467-2.

Gebre, M.T., Hwang, J. and Biru, G. (2024) ‘Electricity demand analysis and forecasting: The case of GADA special economic zone’, *Heliyon*, 10(3), pp. e25364–e25364. Available at: https://doi.org/10.1016/j.heliyon.2024.e25364.

Oliveira, J.M. and Ramos, P. (2024) ‘Evaluating the Effectiveness of Time Series Transformers for Demand Forecasting in Retail’, *Mathematics*, 12(17), p. 2728\. Available at: https://doi.org/10.3390/math12172728.

Rondón-Cordero, V.H., Montuori, L., Alcázar-Ortega, M. and Siano, P. (2025) ‘Advancements in hybrid and ensemble ML models for energy consumption forecasting: results and challenges of their applications’, *Renewable and Sustainable Energy Reviews*, 224, p. 116095\. Available at: https://doi.org/10.1016/j.rser.2025.116095.

Zare, M.S., Nikoo, M.R., Chen, M. and Gandomi, A.H. (2025) ‘Capturing complex electricity load patterns: A hybrid deep learning approach with proposed external-convolution attention’, *Energy Strategy Reviews*, 57, pp. 101638–101638. Available at: https://doi.org/10.1016/j.esr.2025.101638.

	