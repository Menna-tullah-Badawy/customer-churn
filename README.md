# 📉 Customer Churn Intelligence — 1M‑Row End‑to‑End ML Benchmark

> A complete, leakage‑safe machine‑learning study on a **1,000,000‑customer telecom dataset**: EDA → cleaning → **17 regression models**, **21 classification models & ensembles**, **8 clustering algorithms** → business‑cost threshold selection → validation‑driven final model → one‑shot evaluation on an untouched test set → **interactive Gradio app**.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-final%20model-2E8B57)
![XGBoost](https://img.shields.io/badge/XGBoost-benchmarked-EB5E28)
![CatBoost](https://img.shields.io/badge/CatBoost-benchmarked-FFCC00)
![Gradio](https://img.shields.io/badge/Gradio-demo%20app-F97316)

---

## 🎯 Objectives

| Task | Target | Goal |
|---|---|---|
| **Classification** | `churn` (binary, 9.92 % positive) | Identify customers likely to leave, with a decision threshold chosen on *business cost*, not accuracy |
| **Regression** | `totalcharges` | Predict lifetime billing to quantify the revenue at risk |
| **Segmentation** | — (unsupervised) | Discover customer segments and profile them for targeted retention |

---

## 📦 Dataset

* **Source:** Kaggle — [`isandeep06/customer-churn-prediction-dataset-1m`](https://www.kaggle.com/datasets/isandeep06/customer-churn-prediction-dataset-1m) (`customer_churn_1M.csv`, 170 MB)
* **Size:** 1,000,000 rows × 32 columns → 31 modelling features after dropping identifiers (`customer_id`, `signup_date`)
* **Class balance:** 900,773 stay / 99,227 churn → **1 : 9.1 imbalance**
* **Feature groups:** demographics (age, gender, education, marital status, dependents, income), account (tenure, contract, payment method, paperless billing), services (phone, internet, security, backup, streaming…), billing (monthly / total charges, late payments), engagement (satisfaction, complaints, service calls, days since last interaction, credit score)
* **Data quality:** ~3 % missing in `annual_income`, `num_complaints`, ~2 % in `customer_satisfaction`; no duplicates

---

## 🧭 Notebook walkthrough (24 cells)

| # | Stage | What happens |
|---|---|---|
| 1–2 | **Setup & load** | Global config with `FAST_MODE` sampling controls for 1M+ rows; CSV auto‑discovery |
| 3 | **Inspection & memory optimisation** | dtype down‑casting + categorical conversion; missing / duplicate / cardinality report |
| 4 | **Target & leakage detection** | Auto‑detects the churn and regression targets, drops IDs, constants and leakage‑prone columns |
| 5 | **Cleaning pipeline** | Numeric‑as‑text repair, median / `Unknown` imputation, IQR capping (applied only when outliers < 10 % of rows) |
| 6 | **EDA** | 10+ visualisations: churn distribution, numeric distributions, correlations, categorical churn rates |
| 7 | **Preprocessing** | `ColumnTransformer` pipelines — one for linear models (scaling + one‑hot), one for tree models (ordinal) |
| 8–10 | **Regression benchmark** | 17 models across linear, polynomial and tree/boosting families; Lasso‑based feature selection; overfitting‑gap analysis |
| 11–13 | **Classification benchmark** | Stratified 300k sample, `class_weight='balanced'` / `scale_pos_weight`; 19 models + Soft‑Voting & Stacking ensembles |
| 14–16 | **Diagnostics** | ROC / PR curves, confusion matrices, threshold sweep with **business‑cost function**, built‑in & permutation feature importance |
| 17–21 | **Segmentation** | Scaling + PCA; K‑Means elbow / silhouette; comparison of 8 clustering algorithms on 4 internal metrics; segment business profiling |
| 22 | **Validation protocol** | Train / validation / **untouched test** split, model selection + 3‑fold CV + threshold selection on validation only |
| 23 | **Final model** | Retrain on full training set, evaluate **once** on the untouched test set, export artifacts & executive summary |
| 24 | **Gradio app** | Interactive churn scoring UI over the 29 input features with the selected threshold |

---

## 📊 Results

### Regression — predicting `totalcharges`

| Model | R² | RMSE | MAE |
|---|---|---|---|
| **HistGradientBoosting (best)** | **0.9707** | **278.8** | **155.9** |
| CatBoost | 0.9707 | 278.7 | – |
| LightGBM | 0.9706 | 279.5 | – |
| XGBoost | 0.9704 | 280.2 | – |
| Extra Trees / Random Forest | 0.9703 / 0.9702 | 280.5 / 281.0 | – |
| Polynomial (deg 2) + Ridge | 0.9572 | 337.0 | – |
| Linear / Ridge / Lasso / ElasticNet | 0.9252 | 445.5 | – |

Lasso zeroed 4 of 43 engineered features; tree ensembles capture the non‑linear billing structure that linear models miss (+4.5 pts R²).

### Classification — predicting `churn` (final model on the untouched 60,000‑row test set)

| Metric | @ default 0.50 | **@ validation‑selected 0.57** |
|---|---|---|
| Precision | 0.162 | **0.192** |
| Recall | 0.610 | **0.445** |
| F1 | 0.256 | **0.268** |
| ROC‑AUC | 0.682 | 0.682 |
| PR‑AUC | 0.199 | 0.199 |
| MCC | 0.162 | 0.169 |
| Accuracy | 0.648 | 0.759 |

* **Final model:** LightGBM (selected on validation; 3‑fold CV F1 = 0.247 ± 0.002)
* **Confusion matrix @ 0.57:** TN 42,872 · FP 11,174 · FN 3,305 · TP 2,649
* **Business reading:** flagging the top ~23 % of customers captures **44.5 % of all churners** at **≈1.9× lift** over random targeting (precision 0.19 vs. 0.099 base rate).

> **Honest note on the ceiling.** Across 21 classifiers the best ROC‑AUC is ~0.69, and a class‑weighted logistic regression (AUC 0.686) performs on par with tuned gradient boosting (0.679–0.684). When linear and non‑linear models converge like this, the limit is the *information in the features*, not the algorithm — so the notebook invests in what actually moves the business outcome: cost‑aware thresholding, calibrated expectations and a leakage‑free evaluation rather than chasing a headline metric.

### Key churn drivers

| Rank | Feature | Built‑in importance | Permutation ΔF1 |
|---|---|---|---|
| 1 | `contract` | 0.261 | 0.049 |
| 2 | `customer_satisfaction` | 0.108 | 0.010 |
| 3 | `num_service_calls` | 0.062 | 0.007 |
| 4 | `num_complaints` | 0.061 | 0.009 |
| 5 | `has_tech_support` | 0.040 | 0.016 |
| 6 | `late_payments` | 0.030 | 0.011 |

Contract type dominates; service friction (complaints, service calls, missing tech support) and payment behaviour follow — a coherent, actionable story for a retention team.

### Segmentation

8 algorithms were compared (K‑Means, MiniBatch K‑Means, Ward / average / complete hierarchical, DBSCAN, HDBSCAN, GMM, BIRCH) on Silhouette, Davies–Bouldin, Calinski–Harabasz and Dunn index. Silhouette analysis selected **k = 2**; K‑Means was used for profiling:

| Segment | Share | Churn rate | Avg. monthly charges | Avg. total charges | Profile |
|---|---|---|---|---|---|
| **Low‑Value At‑Risk** | 15.1 % | **11.3 %** | $60.7 | $1,296 | No internet / add‑on services, lower spend, higher churn |
| **High‑Value Loyal** | 84.9 % | 9.7 % | $90.4 | $1,867 | Internet + security/backup add‑ons, higher spend |

---

## 🔬 Methodology highlights

* **Leakage‑safe protocol** — identifiers and leakage‑prone columns removed up front; preprocessing fitted inside pipelines; model *and* threshold selected on a validation split; test set touched exactly once.
* **Imbalance handling** — stratified sampling, `class_weight='balanced'` for linear/tree models, `scale_pos_weight ≈ 9.1` for boosting; SMOTE available as an alternative.
* **Cost‑aware decisions** — threshold sweep minimising a false‑negative‑weighted business cost, reported separately from the F1‑optimal threshold, and explicitly marked *exploratory* until re‑selected on validation data.
* **Scalability controls** — `FAST_MODE`, per‑family sample caps (SVM/KNN 40k, GradientBoosting 120k, clustering 60k, hierarchical 8k) and memory down‑casting keep a 1M‑row study runnable on a single Kaggle session.
* **Interpretability** — built‑in and permutation importance on the original feature space (SHAP intentionally skipped for cost).

---

## 📁 Artifacts produced

```text
best_classification_model.pkl      # LightGBM pipeline (threshold 0.57)
best_regression_model.pkl          # HistGradientBoosting pipeline
segmentation_model.pkl             # K-Means (k=2) + scaler/PCA
project_summary.json               # Executive summary of the run
validation_model_comparison.csv    # Validation leaderboard
validation_threshold_analysis.csv  # Threshold sweep on validation data
clustering_comparison.csv          # 8-algorithm clustering benchmark
customer_segments.csv              # Cluster label per customer
segment_profiles.csv               # Business profile per segment
```

---

## 🚀 How to run

1. Open `customer-churn.ipynb` on **Kaggle** and attach the dataset `isandeep06/customer-churn-prediction-dataset-1m` (the loader auto‑discovers the CSV under `/kaggle/input`).
2. Optional — adjust the config block in cell 1:
   ```python
   FAST_MODE      = True      # False = train on the full 1M rows
   MAX_TRAIN_ROWS = 300_000   # stratified sample used for modelling
   TEST_SIZE      = 0.2
   ```
3. Run all cells. The last cell launches the **Gradio** churn‑scoring app (29 inputs → churn probability + decision at the selected threshold).

Local run:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn lightgbm xgboost catboost imbalanced-learn hdbscan joblib gradio
jupyter notebook customer-churn.ipynb   # point DATA_PATH at your local CSV
```

---

## 🧰 Tech stack

`pandas` · `numpy` · `scikit-learn` (pipelines, ColumnTransformer, 25+ estimators, permutation importance) · `LightGBM` · `XGBoost` · `CatBoost` · `imbalanced-learn` · `hdbscan` · `matplotlib` / `seaborn` · `joblib` · `Gradio`

---

## 🗺️ Limitations & next steps

* Feature signal caps ROC‑AUC near 0.69 — next gains should come from **new data** (usage time‑series, support tickets, NPS text) rather than model tuning.
* Add **probability calibration** (isotonic / Platt) and a calibration curve before using scores for cost‑based targeting.
* Replace the sampled benchmark with a **full‑data LightGBM + Optuna** run once the feature set is enriched.
* Package the final pipeline behind a **FastAPI** endpoint with input schema validation, and add drift monitoring on the top drivers.

---

## 👤 Author

**Menna‑tullah Badawy** — [GitHub](https://github.com/Menna-tullah-Badawy)
