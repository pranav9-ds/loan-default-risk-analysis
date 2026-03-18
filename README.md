# Loan Default Risk Analysis
## Loan Default Risk Analysis and Prediction

An end-to-end credit risk analysis on a simulated 601-loan portfolio: from raw data exploration through statistical testing, policy simulation, and tuned machine learning models (Logistic Regression, Random Forest, XGBoost) with SHAP explanations

> **Portfolio project** using simulated loan data. The business scenario is fictional, but the methodology — segment analysis, statistical testing, interaction heatmaps, graduated policy simulation, and ML modeling with SHAP — reflects a real-world credit risk workflow.

---

## Problem Statement

A simulated personal loan portfolio is defaulting at **24.3%** — more than double the **12% target**. The goal is to determine *why* defaults are elevated, *what policy changes* would bring the rate back to target, and *build a model* that predicts default risk for new applications.

## Key Findings

| # | Finding | Evidence |
|---|---------|----------|
| 1 | **Credit score is the #1 risk driver** | 520–599 bucket defaults at 49.1% vs 11.7% for 750+; χ² p < 0.001 |
| 2 | **DTI ratio compounds credit risk** | Score 520–599 + DTI 50–75% → **60.4% default rate** |
| 3 | **Employment tenure < 2 years matters** | 35.1% vs 22.7% default rate; t-test p = 0.038 |
| 4 | **Loan amount is irrelevant** | $22,571 vs $22,013; t-test p = 0.657 |
| 5 | **Employment type does NOT matter** | Part-Time ≈ Full-Time; χ² p = 0.967 |
| 6 | **Late loans are a hidden bomb** | If 103 Late loans convert, portfolio hits 41.4% |

## Recommended Policy

The **Standard policy (credit score ≥ 620, DTI ≤ 55%)** was identified as the optimal trade-off:

| Scenario | Default Rate | Loans Approved | Rejection Rate |
|----------|:-----------:|:--------------:|:--------------:|
| Current (no policy) | 24.3% | 601 | 0% |
| **Standard: CS≥620, DTI≤55** | **12.3%** | **302** | **49.8%** |
| Strict: CS≥650, DTI≤50 | 12.1% | 231 | 61.6% |

Standard achieves essentially the same default rate as Strict (12.3% vs 12.1%) while approving 71 more loans — a 31% volume advantage for 0.2pp of additional risk.

---

## Project Structure

```
loan-default-risk-analysis/
├── data/
│   ├── borrower_profiles.csv                  # 500 borrower demographic records
│   ├── loan_applications.csv                  # 601 loan records with default status
│                 # Merged + feature-engineered dataset
├── notebooks/
│   ├── loan_default_risk_analysis.ipynb       # Part 1: EDA (83 cells)
│   └── loan_default_prediction_model.ipynb
|   └── engineered_loan_data.csv        # Part 2: ML Modeling (38 cells)
├── README.md
└── requirements.txt
```

---

## Part 1 — Exploratory Data Analysis

| Step | Description | Key Output |
|------|-------------|------------|
| 1–2 | Data loading, exploration, cleaning plan | 6 data quality issues documented |
| 3 | Data preparation — merge, flag Late loans, flag extreme DTI | Merged dataset: 601 × 31 |
| 4 | Feature engineering — 5 bucketed segments | credit_score_bucket, dti_range, tenure_group, loan_amount_range, income_bracket |
| 5 | Segment default rate analysis | Default rates across 7 dimensions + time trend |
| 6 | Interaction analysis — Credit Score × DTI heatmap | 60.4% default in worst cell |
| 7 | Statistical testing — t-tests, chi-square | 5 significant, 3 not significant |
| 8 | 17 publication-quality visualizations | Bar charts, heatmaps, violin plots, correlation matrix |
| 9 | Sensitivity analysis — Late loans as default | Portfolio rate bounds: 24.3%–41.4% |
| 10 | Business insights summary | 8 key findings with statistical backing |
| 11 | Graduated policy simulation | 6 scenarios with trade-off curve |
| 12 | Expected loss analysis | $3.3M total loss; 520–599 bucket = 41.8% of all losses |
| 13 | Baseline modeling — LR & RF | LR: CV AUC 0.679, Recall 62%; RF: CV AUC 0.706 |
| 14 | Conclusion, limitations, next steps | Actionable recommendations |

---

## Part 2 — Prediction Model

| Step | Description | Key Output |
|------|-------------|------------|
| 1 | Load engineered dataset from EDA | 601 × 31, 24.3% default rate |
| 2 | Feature selection guided by EDA statistical tests | 7 numeric + 3 categorical → 22 after encoding; `education_level` excluded (p = 0.44) |
| 3 | Baseline models — LR and RF with pipelines | Benchmarks for XGBoost comparison |
| 4 | XGBoost with RandomizedSearchCV (80 iterations, 5-fold) | Tuned hyperparameters, CV-optimized |
| 5 | Model comparison — ROC, PR curves, confusion matrices, CV stability | 3-model comparison across 7 metrics |
| 6 | SHAP explanations — global + individual | Summary plot, bar plot, dependence plots, waterfall plots |
| 7 | Threshold optimization — F1, recall, business impact | Optimal threshold + cost-based simulation |
| 8 | Final selection and conclusions | Recommended model with rationale |

### Model Results

| Metric | Logistic Regression | Random Forest | XGBoost (Tuned) |
|--------|:-------------------:|:-------------:|:---------------:|
| CV AUC (5-fold) | 0.679 ± 0.027 | **0.706 ± 0.050** | 0.698 ± 0.044 |
| ROC-AUC (holdout) | 0.661 | 0.714 | 0.736 |
| Avg Precision | 0.393 | **0.442** | 0.427 |
| Recall (defaults) | **62.2%** | 54.1% | 51.4% |
| Precision (defaults) | 37.7% | **43.5%** | 41.3% |
| F1 (defaults) | 0.469 | **0.482** | 0.458 |

**Recommended model: Random Forest** — highest CV AUC (0.706), highest F1 (0.482), and its holdout AUC (0.714) closely matches its CV mean, indicating stable generalization. XGBoost's holdout AUC (0.736) exceeds its CV mean by +3.8pp, suggesting the test split was favorable rather than indicating genuine superiority.

### Notable Modeling Decisions

- **`education_level` excluded** after chi-square test found no significance (p = 0.44); removing it improved LR AUC from 0.636 → 0.661 and eliminated reference category artifacts
- **Stratified split + stratified 5-fold CV** for stable estimates on a 601-row dataset
- **Pipeline architecture** (`ColumnTransformer` + `StandardScaler` + `OneHotEncoder`) prevents data leakage
- **XGBoost holdout-vs-CV gap flagged** — the notebook warns when holdout AUC exceeds CV mean by >3pp
- **SHAP dependence plots** confirm the Credit Score × DTI interaction discovered in the EDA heatmap
- **Threshold optimization** at 0.49 maximizes F1 (0.518); at 0.30 threshold, recall reaches 91.9% (misses only 3 of 37 defaults)
- **Probability calibration note** — `scale_pos_weight` shifts all probabilities upward; Platt scaling recommended for production

---

## Visualizations (Selected)

The notebooks produce 25+ charts including:

**EDA (Part 1):**
- Default Rate by Credit Score Bucket — bar chart with 12% target line
- Credit Score × DTI Interaction Heatmap — the toxic combination
- Policy Trade-off Curve — default rate vs rejection rate across 6 scenarios
- Sensitivity Analysis — primary vs worst-case by credit score bucket
- Monthly Default Rate Trend — year-month granularity with volume overlay

**Modeling (Part 2):**
- ROC Curves — 3-model comparison (LR, RF, XGBoost)
- Precision-Recall Curves — with prevalence baseline
- Confusion Matrices — side-by-side heatmaps for all 3 models
- CV Stability Boxplot — fold-level AUC variance comparison
- SHAP Summary Plot — global feature importance with direction
- SHAP Dependence Plots — credit score and DTI with interaction coloring
- SHAP Waterfall — individual loan explanations (highest-risk + lowest-risk)
- Threshold Optimization — precision, recall, F1 across all thresholds

---

## Tools & Libraries

| Library | Purpose |
|---------|---------|
| Pandas / NumPy | Data manipulation and feature engineering |
| Matplotlib / Seaborn | 28 publication-quality visualizations |
| SciPy | Welch's t-tests, chi-square tests |
| Scikit-learn | Logistic Regression, Random Forest, Pipeline, StratifiedKFold, metrics |
| XGBoost | Gradient boosted trees with RandomizedSearchCV tuning |
| SHAP | TreeExplainer for exact Shapley values — global and individual explanations |

## How to Run

```bash
git clone https://github.com/Pranava31/loan-default-risk-analysis.git
cd loan-default-risk-analysis
pip install -r requirements.txt

# Part 1: EDA (run first — generates engineered_loan_data.csv)
jupyter notebook notebooks/loan_default_risk_analysis.ipynb

# Part 2: Modeling (requires engineered_loan_data.csv from Part 1)
jupyter notebook notebooks/loan_default_prediction_model.ipynb
```

## Limitations

- **601 samples** — sufficient for EDA and segment analysis; tight for ML (CV variance is inherent)
- **No causal inference** — associations identified, not causes
- **No temporal validation** — ideally train on 2024, test on 2025
- **LGD = 100%** assumed in expected loss — actual recovery rates would reduce estimates
- **Late loan outcomes unknown** — sensitivity analysis bounded the range
- **XGBoost holdout may be optimistic** — +3.8pp above CV mean on small dataset

---

*Built by Pranav — MS Information Systems, Northeastern University*
