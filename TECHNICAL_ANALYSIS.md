# Technical Analysis — Car Price Prediction Pipeline

> Full methodology, engineering decisions, and experimental findings across all 8 notebooks.

---

## 1. Problem Definition

**Task:** Regression — predict the resale price (USD) of a used car from structured listing attributes.

**Target variable:** `priceUSD` (continuous, right-skewed)

**Primary metric:** RMSE (Root Mean Squared Error)
**Secondary metric:** MAE (Mean Absolute Error)
**Excluded metric:** R² — see [Section 2.4](#24-why-r²-is-excluded)

**Data source:** Used car listings dataset — 56,244 rows, 12 columns, predominantly Eastern European market listings.

---

## 2. Evaluation Methodology

### 2.1 Why RMSE as Primary Metric

RMSE penalises large prediction errors more heavily than MAE because residuals are squared before averaging. For car price prediction, this is critical:

- A **$10,000 error on a $50,000 luxury car** is a business failure — the customer would reject the valuation entirely
- Ten **$1,000 errors on $5,000 budget cars** are individually manageable
- MAE treats both scenarios identically; RMSE correctly weights the luxury car mistake as far more severe

RMSE also naturally amplifies the signal from the skewed price tail — luxury cars are rare in the training set, and RMSE ensures the model is penalised disproportionately when it gets them badly wrong.

### 2.2 MAE as Secondary Metric

MAE is reported alongside RMSE for interpretability: a MAE of $1,162 means the model is off by $1,162 on average across all predictions. This is easy to communicate to non-technical stakeholders.

### 2.3 RMSE/MAE Ratio as Diagnostic

The ratio `RMSE / MAE` indicates the severity of tail errors:
- Ratio ≈ 1.0 → errors are uniformly distributed (no outliers)
- Ratio > 1.5 → model is making a few very large errors on outlier cars

For our final model: $2,515 / $1,162 ≈ **2.17×** — confirms that while average errors are manageable, the model struggles at the luxury end of the distribution.

### 2.4 Why R² is Excluded

R² measures what proportion of the *variance* in price the model explains. On our dataset, price ranges from ~$250 to ~$190,141 with extreme right skew — the variance is dominated by a handful of luxury cars.

A model that gets the $100,000+ cars approximately right will score R² > 0.90 even if it makes $3,000 errors on the thousands of $5,000 cars that dominate the dataset. R² rewards variance capture, not prediction accuracy. We excluded it entirely from model selection to avoid being misled.

### 2.5 Train / Test Split Design

- **Split ratio:** 80% train / 20% test (44,906 / 11,227 rows)
- **Strategy:** Stratified on `price_category` (budget / mid / luxury — 33rd and 66th percentile bins)
- **Random state:** 42 (reproducible across Databricks and Colab)
- **Why stratify:** The price distribution is right-skewed. Without stratification, random splitting could produce a test set dominated by cheap cars, giving an overly optimistic RMSE. Stratification guarantees equal representation of all price tiers in both sets.
- **`price_category` handling:** The column is created in NB04 purely for stratification, used in NB05 (Databricks) and NB08 (Colab) for the split, then dropped from the feature matrix — it is never a training input.

### 2.6 Overfitting Detection

A model is flagged as overfitting if the **test MAE exceeds train MAE by more than 20%**. MAE is used for this check (not RMSE) because RMSE is sensitive to a few large errors on luxury cars, which can inflate the train/test gap even when the model generalises well. MAE gives a more stable and fair overfitting signal. This threshold was chosen pragmatically — a small generalisation gap is expected and healthy; 20%+ indicates memorisation rather than learning.

---

## 3. Data Pipeline — Databricks (NB01–NB05)

### 3.1 NB01 — Data Gathering

Raw `cars.csv` (56,244 rows × 12 columns) is loaded via pandas and saved to the Databricks Unity Catalog as `workspace.default.carprice_project_raw_data`. No transformations applied at this stage.

**Raw columns:** `make`, `model`, `priceUSD`, `year`, `condition`, `mileage(kilometers)`, `fuel_type`, `volume(cm3)`, `color`, `transmission`, `drive_unit`, `segment`

### 3.2 NB02 — Data Cleaning

**Design principle:** Only rule-based removals are done before the train/test split. No statistical transformations (imputation, outlier removal) are applied here — those require train statistics and must happen after splitting to prevent leakage.

Cleaning steps:
1. **Column renaming:** Remove special characters from column names (`mileage(kilometers)` → `mileage_kilometers`, `volume(cm3)` → `volume_cm3`)
2. **Text standardisation:** Lowercase all string values for consistency
3. **Corrupt data removal (rule-based):**
   - Mileage = 0 for non-electric cars (physically impossible)
   - Price = 100 0r less 
   - Exact duplicate rows
4. **Missing value reporting:** Identified and documented, but NOT imputed here

**Output:** 56,133 rows × 11 columns (111 rows removed, 0.2% loss — the data is very clean)

### 3.3 NB03 — Exploratory Data Analysis

Read-only analysis — no output table. Key findings that informed downstream decisions:

| Finding | Design Decision Made |
|---|---|
| `model` has very high cardinality (thousands of unique values) | Dropped in NB04 — too sparse after OHE |
| `make` has ~50 unique values with many rare makes | Grouped into `make_category` in NB04 |
| `year` correlates strongly with price (r ≈ +0.6) | Replaced by `car_age` in NB04 |
| `mileage_km` has negative correlation with price | Kept as-is, also used in `mileage_per_year` |
| Price is heavily right-skewed (mean >> median) | Chose RMSE over MAE; rejected log-transform |
| `volume_cm3` has 47 missing values (all EVs) | Impute with 0 (genuine zero, not missing) in NB05 |

**EDA insights on price drivers:**
- **Strongest features:** `car_age`, `mileage_km`, `make_category`, `segment`
- **Secondary features:** `fuel_type`, `transmission`, `drive_unit`, `condition`, `volume_cm3`, `color`
- **Multicollinearity check:** `year` and `car_age` would be perfectly collinear — one must be dropped

### 3.4 NB04 — Feature Engineering

**New features created:**

| Feature | Formula | Rationale |
|---|---|---|
| `car_age` | 2019 − year | Direct depreciation proxy — avoids raw year values |
| `mileage_per_year` | mileage_km ÷ car_age | Usage intensity — distinguishes high-mileage new cars from low-mileage old cars |
| `make_category` | Group makes with < 100 listings into 'other' | Prevents OHE from creating hundreds of sparse dummy columns for rare brands |
| `price_category` | 33rd/66th percentile bins → budget/mid/luxury | Stratification key only — dropped before training |

**Columns dropped:**

| Column | Reason |
|---|---|
| `model` | Very high cardinality — too sparse for encoding |
| `year` | Replaced by `car_age` (more interpretable, avoids multicollinearity) |
| `make` | Replaced by `make_category` |

### 3.5 NB05 — Preprocessing

**All preprocessing is fit on training data only** to prevent data leakage into the test set.

| Step | Method | Details |
|---|---|---|
| Train/test split | Stratified (price_category) | 80/20, random_state=42 |
| `volume_cm3` imputation | Fill with 0 | 47 missing values = EVs (no combustion engine → displacement is genuinely zero, not unknown) |
| Other numeric imputation | Median (train only) | Robust to outliers |
| Categorical imputation | Mode (train only) | Fill rare missing categoricals |
| Numeric scaling | StandardScaler (fit on train) | Normalises to mean=0, std=1 — required for LR coefficient stability |
| Categorical encoding | OneHotEncoder(drop='first') | Avoids the dummy variable trap — without `drop='first'`, all OHE columns for a group sum to 1, creating perfect multicollinearity that causes LR coefficients to explode to arbitrary values |

**Why `drop='first'`:** Without dropping one reference category per group, all binary columns for a categorical feature (e.g., all `transmission_*` columns) are perfectly collinear — they sum to exactly 1 for every row. Linear Regression has no unique solution in this case and assigns meaningless coefficients of quadrillions. Dropping the first category breaks this collinearity.

**Why 0 for EV `volume_cm3`:** Imputing with the median (~1,600 cm³) would incorrectly signal to models that electric vehicles have a combustion engine. The physical displacement of an EV is genuinely zero, not missing.

**Output:** `workspace.default.carprice_project_preprocessed_data` (combined train + test with `split` column)

---

## 4. Model Training — Databricks (NB06)

### 4.1 Evaluation Function Design

A shared `evaluate_model()` function is used for all models:
- Computes MAE and RMSE on both train and test sets
- Prints RMSE with `(*)` label denoting primary metric
- Reports R² for reference only (explicitly not used for ranking)
- Flags overfitting if test MAE > train MAE × 1.20 (uses MAE gap — more stable than RMSE on skewed price distributions)

### 4.2 Linear Regression — Baseline

Linear Regression serves as the interpretability baseline — its coefficients reveal what the model learned about car pricing factors.

**Results:**
- Train RMSE: $5,548 | Test RMSE: $5,235
- Train MAE: $2,656 | Test MAE: $2,607
- No overfitting (MAE gap: −$49, well within 20% threshold)

**Coefficient interpretation (quantitative features, per standard deviation):**
- `car_age`: −$3,667 per std dev (~−$312/year — std dev ≈ 11.7 years)
- `volume_cm3`: +$596 per std dev
- `mileage_km`: −$1,362 per std dev
- `mileage_per_year`: +$1,145 per std dev (positive due to multicollinearity — see note below)

> **Note on `mileage_per_year` being positive:** Once LR controls for total `mileage_km`, a higher `mileage_per_year` signals a newer car driven more recently — and newer cars are worth more. The two mileage features partially cancel each other in a linear model. Tree models handle this more naturally.

**Coefficient interpretation (dummy variables — direct price effect):**
- `fuel_type_electrocar`: +$22,837 (EV premium)
- `make_category_porsche`: +$10,533
- `make_category_gaz`: +$8,518
- `drive_unit_rear_drive`: −$4,739 (multicollinearity with premium brands)
- `drive_unit_front_wheel_drive`: −$4,685
- `make_category_ssangyong`: −$3,293

### 4.3 Random Forest — Ensemble Baseline

Initial training produced severe overfitting (train MAE $445 vs test MAE $1,142 — a 157% gap). Regularisation was applied:

| Parameter | Value | Purpose |
|---|---|---|
| `max_depth` | 15 | Prevents trees from memorising individual training examples |
| `min_samples_split` | 10 | Node must have ≥ 10 samples before splitting |
| `min_samples_leaf` | 5 | Each leaf must contain ≥ 5 samples |
| `max_features` | 'sqrt' | Each split considers √(n_features) features |

**After regularisation:** Train RMSE $3,446, Test RMSE $3,382 | Train MAE $1,512, Test MAE $1,539 — overfitting eliminated (MAE gap: +$27, well within 20% threshold). However, regularisation also limited performance: XGBoost with default settings outperformed.

### 4.4 XGBoost — Best Manual Model

XGBoost uses sequential boosting: each tree corrects the errors of all previous trees. This makes it more efficient than Random Forest at targeting the hardest-to-predict residuals.

**Configuration:** `n_estimators=100`, `learning_rate=0.1`, `random_state=42`

**Results:**
- Train RMSE: $2,098 | Test RMSE: $2,551
- Train MAE: $1,159 | Test MAE: $1,219
- Overfitting gap: +$60 MAE (+5.2%) — healthy, well within 20% threshold

**Boosting round analysis:** MAE drops steeply in rounds 1–30 (model captures main patterns: `car_age`, `volume_cm3`), then flattens after ~60–80 rounds. Additional trees give diminishing returns — 100 trees is sufficient.

**RF vs XGBoost comparison:**

| Metric | Random Forest | XGBoost | Winner |
|---|---|---|---|
| Test RMSE | $3,382 | $2,551 | XGBoost (by $831) |
| Test MAE | $1,539 | $1,219 | XGBoost (by $320) |
| Overfitting | No | No | Both pass |

### 4.5 Decision Tree — Interpretability Reference

A single decision tree is included to demonstrate the tradeoff between interpretability and accuracy. Constrained to `max_depth=10` to prevent severe overfitting, but performance is significantly below ensemble methods.

### 4.6 MLflow Tracking

All 4 models are logged to a single MLflow experiment on Databricks:
- **Logged metrics:** `train_mae`, `test_mae`, `train_rmse`, `test_rmse`, `mae_gap`, `train_r2`, `test_r2`
- **Primary selection metric:** `test_rmse`
- R² logged for reference only

---

## 5. Prediction & Evaluation — Databricks (NB07)

NB07 loads the trained models from MLflow and performs final evaluation and interpretability analysis.

### 5.1 Final Manual Model Comparison

| Model | Test RMSE (Primary) | Test MAE | RMSE Gap | Overfitting |
|---|---|---|---|---|
| Linear Regression | $5,235 | $2,607 | — | No |
| Random Forest | $3,382 | $1,539 | — | No |
| XGBoost | $2,551 | $1,219 | +$453 | No |
| Decision Tree | Higher than RF | — | — | No |

**Best model: XGBoost** (lowest test RMSE and test MAE, no overfitting)

### 5.2 Residual Analysis

XGBoost residuals are approximately normally distributed and centred near zero, confirming the model is unbiased. The right tail of the residual distribution is heavier than the left — consistent with the known difficulty of predicting luxury car prices accurately with limited training samples.

### 5.3 Coefficient Analysis (Linear Regression)

The LR coefficient analysis (NB07 Section 7b) provides a fully interpretable decomposition of price drivers — even though LR is not the best model, its transparency makes it valuable for business communication:
- Depreciation is the dominant effect (~−$312/year per year of age)
- Electric vehicles command a $22,837 premium over comparable petrol cars
- Brand premiums range from +$10,533 (Porsche) to −$3,293 (SsangYong)
- Drive unit discounts: rear-drive −$4,739, front-wheel drive −$4,685 (multicollinearity with premium brands)

---

## 6. AutoML Extension — Google Colab (NB08)

### 6.1 Featuretools Deep Feature Synthesis

DFS automatically generates features by applying mathematical primitives to numeric columns:

**Primitives applied:** `add_numeric`, `subtract_numeric`, `multiply_numeric`, `divide_numeric`, `natural_logarithm`, `square_root`

**Input columns:** `year`, `mileage_kilometers`, `volume_cm3` (3 numeric columns)

**Output:** 27 numeric features (pairwise combinations + individual transformations)

**Why DFS on numerics only:** Featuretools `trans_primitives` only operate on numeric columns. Categorical columns were excluded from the EntitySet and rejoined manually after DFS.

**Total feature matrix:** 34 features before OHE → **100 features after OHE** (27 numeric + 73 binary columns from 7 categorical features)

### 6.2 PyCaret AutoML

**Setup:**
- Training set only passed to `setup()` — test set held out entirely
- 5-fold cross-validation on training data
- `session_id=42` for reproducibility
- Sort metric: RMSE (primary metric)
- `n_select=20` — keeps top 20 models

**`compare_models()` results (top 5 by CV RMSE):**

| Rank | Model | CV RMSE | CV MAE |
|---|---|---|---|
| 1 | CatBoost | $2,750 | $1,211 |
| 2 | XGBoost | $2,959 | $1,238 |
| 3 | Extra Trees | $3,000 | $1,255 |
| 4 | Random Forest | $3,071 | $1,255 |
| 5 | LightGBM | $3,041 | $1,317 |

All top 4 models are tree-based ensembles — confirming the car price prediction problem is non-linear and benefits from ensemble methods.

**`tune_model()` (Random Search, 50 iterations, optimize=RMSE):**
All 50 hyperparameter configurations produced higher CV error than the defaults. PyCaret automatically reverted to the original CatBoost configuration. This is expected — CatBoost's defaults are already well-optimised for tabular data.

**`finalize_model()`:**
Retrains the best configuration on 100% of the training data (44,906 rows vs ~35,924 during CV folds). More training data → better generalisation → lower test error.

**Final CatBoost results:**

| Split | RMSE | MAE |
|---|---|---|
| Train | $1,714 | $1,031 |
| Test | $2,515 | $1,162 |
| Gap | +$801 | +$131 |

### 6.3 Why CatBoost Wins

CatBoost (Categorical Boosting) differs from XGBoost in several key ways relevant to this dataset:
1. **Native categorical handling:** CatBoost handles categorical features internally without OHE, reducing information loss from encoding
2. **Ordered boosting:** Reduces target leakage during training by using different subsets for computing leaf values vs tree structure
3. **Symmetric trees:** All nodes at the same depth use the same split condition — reduces overfitting on tabular data with many categorical features

On this dataset with 7 categorical features and a right-skewed target, CatBoost's architecture is better suited than XGBoost's default configuration.

### 6.4 Three-Approach Comparison

| Approach | Feature Engineering | Model Selection | Tuning | Test RMSE |
|---|---|---|---|---|
| Manual (NB06/07) | 3 hand-crafted features | Human selection (3 models) | Manual | $2,551 |
| PyCaret only (hand features) | Same 3 manual features | AutoML (20 models) | Automated | Not directly measured |
| Featuretools + PyCaret | 100 auto-generated features | AutoML (20 models) | Automated | **$2,515** |

Key insight: more features alone did not guarantee a better model — automated model selection (which found CatBoost, never tried in NB06) was the decisive factor.

---

## 7. Model Interpretability

### 7.1 LIME — Local Explanations

LIME explains individual predictions by locally approximating the complex model with a simple linear model around the specific data point.

**Methodology:**
1. Take the target car's feature values
2. Generate hundreds of perturbed samples (same car, varying features)
3. Query the CatBoost model for predictions on all perturbed samples
4. Fit a weighted linear regression on the perturbed samples (weighted by similarity to original)
5. The linear model's coefficients are the LIME explanation

**Three representative test cases:**

| Car | Actual | Predicted | RMSE Contribution | Insight |
|---|---|---|---|---|
| Budget ($214) | $214 | −$1,962 | Very high | Extrapolation failure — below training distribution |
| Mid-range | $4,890 | $5,383 | Low ($493) | Strong performance on mainstream cars |
| Luxury | $190,141 | $119,323 | Very high ($70,818) | Luxury tail underfitting — few training examples |

**LIME actionable findings:**
- **Budget tail:** Apply minimum price floor post-prediction (floor at $0)
- **Luxury tail:** Collect more data above $50k, or train a separate luxury segment model
- **Mid-range:** Model is production-ready for mainstream cars

### 7.2 SHAP — Global and Local Explanations

SHAP uses Shapley values from cooperative game theory to allocate each feature's contribution to every prediction.

**SHAP expected value (baseline):** $7,407 — the average prediction before any features are considered.

Every prediction is decomposed as:
```
predicted_price = $7,407 + SHAP(year) + SHAP(mileage) + SHAP(volume_cm3) + ...
```

**Top global features (SHAP bar chart — mean absolute contribution):**
1. Year/car_age-derived Featuretools features (depreciation)
2. Mileage-derived Featuretools features (usage intensity)
3. `make_category_*` premium brand dummies (brand premium)
4. `segment_*` dummies (vehicle class premium)
5. `volume_cm3` combinations (engine size proxy for luxury)

**Waterfall plots:**
- **Luxury car:** Large positive contributions from brand, segment, and year features pushing prediction far above baseline
- **Budget car:** Large negative contributions from high mileage, older age, and economy brand, pulling prediction well below baseline

**SHAP vs LIME comparison:**

| Aspect | LIME | SHAP |
|---|---|---|
| Scope | Local (one prediction) | Both global and local |
| Consistency | Local approximation only | Mathematically guaranteed additivity |
| Speed | Faster | Slower (requires all predictions) |
| Use case | Explain a specific decision | Understand the model overall |

---

## 8. Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Very few training examples below $500 | Model extrapolates to negative prices at extreme low end | Apply post-prediction price floor |
| Very few training examples above $100,000 | Luxury cars underestimated by 30–40% | More luxury data; separate luxury model |
| `price_category` bins based on percentiles | Budget/mid/luxury definitions are relative, not absolute | Fixed at 33rd/66th percentile of this dataset |
| NB08 replicates NB02+NB04+NB05 manually | Risk of subtle divergence from Databricks pipeline | Verified dataset shapes match at each stage |
| Featuretools DFS on 3 columns only | Limited feature diversity from DFS | All 7 categorical features rejoined manually |
| No cross-validation in manual pipeline (NB06) | Single train/test evaluation may have variance | Mitigated by stratified split and large dataset size |

---

## 9. Reproducibility Notes

| Parameter | Value |
|---|---|
| `random_state` | 42 (all models and splits) |
| `session_id` (PyCaret) | 42 |
| Train/test split | 80/20, stratified on price_category |
| Databricks Unity Catalog | `workspace.default.*` |
| MLflow experiment | Notebook path-based (auto-created) |
| Python version (Colab) | 3.12 (with PyCaret version check patched) |
| PyCaret version | Full install (`pycaret[full]`) |
