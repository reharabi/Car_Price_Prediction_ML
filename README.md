# Car Price Prediction — End-to-End Machine Learning Pipeline

> Predicting second-hand car prices from 56,000+ listings using a fully orchestrated ML pipeline on Databricks, extended with AutoML and model interpretability in Google Colab.

---

## Project Overview

This project builds and evaluates a machine learning pipeline to predict used car prices from structured listing data. The pipeline spans **8 notebooks** across two platforms:

- **Notebooks 1–7 (Databricks):** A fully orchestrated, production-style pipeline — data ingestion, cleaning, EDA, feature engineering, preprocessing, model training, and evaluation — all managed through the **Databricks Unity Catalog**.
- **Notebook 8 (Google Colab):** An AutoML extension using **PyCaret** and **Featuretools** (Deep Feature Synthesis), with **SHAP** and **LIME** model interpretability.

**Primary evaluation metric: RMSE** — chosen because catching large mispredictions (especially on luxury cars) matters more than optimising average error.

---

## Key Results

| Model | Approach | Test RMSE (Primary) | Test MAE |
|---|---|---|---|
| Linear Regression | Manual (NB06/07) | $5,235 | $2,607 |
| Random Forest | Manual (NB06/07) | $3,382 | $1,539 |
| XGBoost (best manual) | Manual (NB06/07) | $2,551 | $1,219 |
| XGBoost (Featuretools) | Automated features (NB08) | $2,810 | $1,248 |
| **PyCaret CatBoost** | **Fully automated (NB08)** | **$2,515** | **$1,162** |

**Winner: PyCaret CatBoost** — beats the best manual model by **$36 RMSE** with zero manual model selection effort.

---

## Repository Structure


CarPrice/
├── [Data Gathering](https://github.com/reharabi/Car_Price_Prediction_ML/blob/main/notebooks/01_data_gathering%20(final).ipynb)          # Load raw CSV into Databricks Unity Catalog
├── [Data Cleaning](https://github.com/reharabi/Car_Price_Prediction_ML/blob/main/notebooks/02_data_cleaning%20(final).ipynb)           # Remove corrupt data, standardise columns
├── [EDA](https://github.com/reharabi/Car_Price_Prediction_ML/blob/main/notebooks/03_eda%20(final).ipynb)                     # Exploratory Data Analysis — distributions & correlations
├── [Feature Engineering](https://github.com/reharabi/Car_Price_Prediction_ML/blob/main/notebooks/04_Feature_engineering%20(final).ipynb)     # Create car_age, mileage_per_year, make_category
├── [Preprocessing](https://github.com/reharabi/Car_Price_Prediction_ML/blob/main/notebooks/05_preprocessing%20(final).ipynb)           # Stratified split, imputation, scaling, OHE
├── [Model Training](https://github.com/reharabi/Car_Price_Prediction_ML/blob/main/notebooks/06_model_training%20(final).ipynb)          # Train LR, RF, XGBoost, Decision Tree — MLflow logging
├── [Prediction Evaluation](https://github.com/reharabi/Car_Price_Prediction_ML/blob/main/notebooks/07_prediction_evaluation%20(final).ipynb)   # Evaluate models, select best, SHAP/residual analysis
├── [Pycaret Comparison](https://github.com/reharabi/Car_Price_Prediction_ML/blob/main/notebooks/08_pycaret_comparison%20(final).ipynb)      # AutoML (PyCaret) + Featuretools + LIME + SHAP (Colab)
├── README.md                        # This file
├── TECHNICAL_ANALYSIS.md            # Full methodology and engineering decisions
└── EXECUTIVE_SUMMARY.md             # Non-technical summary for stakeholders


---

## Pipeline Architecture

### Databricks Pipeline (NB01–NB07) — Orchestrated via Unity Catalog

```
Raw CSV (cars.csv)
      │
      ▼
[NB01] Data Gathering
   └─→ workspace.default.carprice_project_raw_data
      │
      ▼
[NB02] Data Cleaning
   └─→ workspace.default.carprice_project_cleaned_data
      │
      ▼
[NB03] EDA  (read-only, no output table)
      │
      ▼
[NB04] Feature Engineering
   └─→ workspace.default.carprice_project_engineered_data
      │
      ▼
[NB05] Preprocessing  (stratified split + imputation + scaling + OHE)
   └─→ workspace.default.carprice_project_preprocessed_data
      │
      ▼
[NB06] Model Training  (LR · RF · XGBoost · Decision Tree)
   └─→ MLflow experiment (4 model runs logged)
      │
      ▼
[NB07] Prediction & Evaluation
   └─→ workspace.default.carprice_project_evaluation_results
   └─→ Best model: XGBoost  (Test RMSE $2,551 · Test MAE $1,219)
```

Each notebook reads from and writes to the Unity Catalog, forming a traceable, reproducible data lineage. All preprocessing steps are **fit on training data only** to prevent leakage.

### Google Colab Extension (NB08) — AutoML + Interpretability

```
cars.csv  (same raw data, replicated cleaning & feature engineering)
      │
      ▼
Featuretools DFS  →  100 features (27 numeric + 73 OHE)
      │
      ▼
Stratified split  (80/20 · random_state=42 · same strategy as NB05)
      │
      ▼
PyCaret compare_models()  →  20 models benchmarked (5-fold CV, sorted by RMSE)
      │
      ▼
tune_model()  →  50 random configs (all worse → auto-reverted to defaults)
      │
      ▼
finalize_model()  →  retrained on 100% of training data
      │
      ▼
CatBoost  →  Test RMSE $2,515 · Test MAE $1,162
      │
      ├──→ LIME  (3 representative cars: budget · mid-range · luxury)
      └──→ SHAP  (global bar chart + beeswarm + 2 waterfall plots)
```

---

## Dataset

| Property | Value |
|---|---|
| Source | Used car listings (Eastern European market) |
| Raw size | 56,244 rows × 12 columns |
| After cleaning | 56,133 rows × 11 columns (111 rows removed, 0.2%) |
| Price range | ~$250 – $190,141 |
| Distribution | Heavily right-skewed — majority under $10,000 |
| Train / Test split | 80% / 20% (44,906 / 11,227 rows) |
| Split strategy | Stratified on `price_category` (budget / mid / luxury) |

**Raw features:** `make`, `model`, `year`, `condition`, `mileage_km`, `fuel_type`, `volume_cm3`, `color`, `transmission`, `drive_unit`, `segment`, `priceUSD`

---

## Feature Engineering

### Manual (NB04 — Databricks)

| Feature | Description |
|---|---|
| `car_age` | 2019 − year — depreciation proxy |
| `mileage_per_year` | mileage_km ÷ car_age — usage intensity |
| `make_category` | Groups rare makes (< 100 listings) into 'other' |
| `price_category` | Budget / Mid / Luxury bins — for stratified split only, dropped before training |
| Dropped `model` | Very high cardinality — too sparse for encoding |
| Dropped `year` | Replaced by `car_age` |
| Dropped `make` | Replaced by `make_category` |

### Automated (NB08 — Featuretools DFS)

Featuretools Deep Feature Synthesis applied transformation primitives (`add`, `subtract`, `multiply`, `divide`, `log`, `sqrt`) to the 3 raw numeric columns, generating **27 numeric features**. Combined with 7 categorical columns (one-hot encoded), the final feature matrix contains **100 features**.

---

## Models Trained

### Manual Pipeline (NB06)

| Model | Key Config | Test RMSE | Test MAE | Overfitting |
|---|---|---|---|---|
| Linear Regression | Default sklearn | $5,235 | $2,607 | No |
| Random Forest | max_depth=15, min_samples_leaf=5 | $3,382 | $1,539 | No |
| XGBoost | n_estimators=100, lr=0.1 | $2,551 | $1,219 | No |
| Decision Tree | max_depth=10 | Higher than RF | — | No |

All models logged to **MLflow** with train RMSE, test RMSE, train MAE, test MAE, and R² (reference only).

### AutoML Pipeline (NB08 — PyCaret, 20 models)

Top 5 by CV RMSE (5-fold, training set only):

| Rank | Model | CV RMSE | CV MAE |
|---|---|---|---|
| 1 | CatBoost | $2,750 | $1,211 |
| 2 | XGBoost | $2,959 | $1,238 |
| 3 | Extra Trees | $3,000 | $1,255 |
| 4 | Random Forest | $3,071 | $1,255 |
| 5 | LightGBM | $3,041 | $1,317 |

---

## Model Interpretability

### LIME (Local Explanations)
Three representative cars explained from the test set:

| Segment | Actual Price | Predicted | Error |
|---|---|---|---|
| Budget | $214 | −$1,962 | Known extrapolation failure at extreme low end |
| Mid-range | $4,890 | $5,383 | $493 (10%) — strong performance |
| Luxury | $190,141 | $119,323 | $70,818 (37%) — luxury tail underfitting |

Key drivers identified by LIME: `year`, `mileage_km`, `make_category`, `segment`.

**LR key coefficients:** EV premium +$22,837 | Porsche +$10,533 | GAZ +$8,518 | rear-drive −$4,739 | car_age −$312/year

### SHAP (Global + Local)
- **SHAP baseline (expected value):** $7,407
- **Top global features:** `year`-derived features, `mileage_km`-derived features, `make_category_*` dummies, `segment_*` dummies
- **Waterfall plots** generated for the highest and lowest predicted price cars

---

## Why RMSE over MAE?

RMSE penalises large errors more heavily by squaring residuals before averaging. In car pricing:
- A **$10,000 error** on a luxury car is catastrophically worse than ten **$1,000 errors** on budget cars
- RMSE captures that asymmetry; MAE treats every dollar of error equally
- R² is excluded entirely — on a right-skewed price distribution ($250–$190k), R² is artificially inflated by a few very expensive cars, making models look better than they are for everyday mid-range cars

MAE is reported as a secondary metric for interpretability.

---

## Technology Stack

| Tool | Purpose |
|---|---|
| **Databricks** | Pipeline orchestration, Unity Catalog, MLflow |
| **Apache Spark** | Distributed data storage (Unity Catalog tables) |
| **Python / pandas / numpy** | Data manipulation |
| **scikit-learn** | LR, RF, Decision Tree, preprocessing, metrics |
| **XGBoost** | Gradient boosting model |
| **PyCaret** | AutoML — 20-model benchmark + tuning |
| **Featuretools** | Automated feature engineering (Deep Feature Synthesis) |
| **CatBoost** | Best model (via PyCaret AutoML) |
| **SHAP** | Global + local model interpretability |
| **LIME** | Individual prediction explanations |
| **MLflow** | Experiment tracking and model registry |
| **Google Colab** | NB08 runtime environment |
| **matplotlib / seaborn** | Visualisations |

---

## How to Run

### Databricks (NB01–NB07)
1. Upload `cars.csv` to DBFS (`/FileStore/tables/cars.csv`)
2. Run notebooks **in order**: NB01 → NB02 → NB03 → NB04 → NB05 → NB06 → NB07
3. Each notebook reads the output table of the previous one from Unity Catalog

### Google Colab (NB08)
1. Upload `cars.csv` to Google Drive (`/content/drive/MyDrive/cars.csv`)
2. Open `08_pycaret_comparison.ipynb` in Google Colab
3. Run **Step 1** (install dependencies) then **restart the runtime**
4. Run all remaining cells in order

> NB08 replicates NB02 + NB04 + NB05 cleaning and feature engineering steps internally — no dependency on Databricks outputs.

---

## Key Design Decisions

- **No data leakage:** All preprocessing (imputation, scaling, encoding) is fit on training data only, then applied to the test set
- **Stratified splitting:** `price_category` used for stratification then dropped — ensures equal representation of budget, mid-range, and luxury cars in both train and test
- **R² excluded from model selection:** Only RMSE and MAE used for ranking — R² is misleading on skewed distributions
- **Log transform rejected:** Applying log(price) distorts the error metric interpretation; errors become percentage-based rather than dollar-based, which is less actionable
- **MLflow tracking:** All Databricks model runs logged with full metrics for reproducibility and comparison

