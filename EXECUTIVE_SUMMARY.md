# Executive Summary — Car Price Prediction

---

## What We Built

A machine learning system that predicts the resale price of a used car from its listing details — make, age, mileage, fuel type, transmission, and other attributes. Given any car's specifications, the system outputs a predicted price in USD.

The project was built as an end-to-end pipeline, covering every stage from raw data to a deployable model — with rigorous experimentation, automated benchmarking, and transparent model explanations.

---

## The Business Problem

Used car pricing is highly subjective and inconsistent. Sellers overprice or underprice based on intuition, while buyers lack reference points for fair value. A reliable price prediction model enables:

- **Sellers** to list at a competitive, data-backed price
- **Buyers** to identify fair deals and spot overpriced listings
- **Platforms** to automate price suggestions at scale
- **Dealerships** to appraise trade-ins consistently and quickly

The model is especially valuable for avoiding large pricing errors — a $10,000 mistake on a luxury car is far more damaging than a $1,000 mistake on a budget car. This asymmetry shaped every technical decision in the project.

---

## The Data

We worked with **56,244 real used car listings** covering a wide range of makes, models, years, and conditions. After removing a small number of corrupt or duplicate records (just 0.2% of the data), the clean dataset contained **56,133 listings** with prices ranging from approximately **$250 to $190,000**.

The majority of cars are priced under $10,000 — this skewed distribution means predicting luxury cars accurately is genuinely difficult. The model performs well across the mainstream segment and acknowledges its limitations at the extreme ends of the price range.

---

## Our Approach

### Phase 1: Manual Pipeline on Databricks (7 Notebooks)

We built a fully **orchestrated, production-style pipeline** on the Databricks cloud platform. Each step runs as a separate notebook and passes data to the next through a structured data catalog — creating a complete, traceable data lineage from raw CSV to final model.

**The pipeline:**

| Step | What happened |
|---|---|
| **Data Loading** | Raw listings ingested into the Databricks data catalog |
| **Data Cleaning** | Corrupt entries removed (bad mileage, duplicate listings) |
| **Exploration** | Patterns uncovered — age, mileage, brand, and fuel type are the strongest price drivers |
| **Feature Engineering** | New variables created: car age, annual mileage intensity, brand groupings |
| **Preprocessing** | Data split into training and test sets; missing values filled; features scaled and encoded — all without contaminating the test set |
| **Model Training** | Four models trained: Linear Regression, Random Forest, XGBoost, and Decision Tree |
| **Evaluation** | Models compared on held-out test data; XGBoost selected as the best manual model |

### Phase 2: AutoML on Google Colab (1 Notebook)

We extended the project with **PyCaret**, an automated machine learning library that benchmarks up to 20 different models at once — removing human bias from model selection. We also used **Featuretools** to automatically generate 100 features from the raw data, replacing the 3 hand-crafted features from Phase 1.

**The automated pipeline:**
1. 100 features generated automatically
2. 20 models benchmarked in parallel using cross-validation
3. Best model selected, tuned, and retrained on full training data
4. Predictions explained using LIME and SHAP

---

## Results

### Which model won?

| Model | RMSE — Primary Metric | MAE — Secondary |
|---|---|---|
| Linear Regression (manual) | $5,235 | $2,607 |
| Random Forest (manual) | $3,382 | $1,539 |
| XGBoost — best manual model | $2,551 | $1,219 |
| **PyCaret CatBoost — AutoML winner** | **$2,515** | **$1,162** |

**PyCaret CatBoost** is the overall winner with a test **RMSE of $2,515** — our primary metric, chosen because it penalises large luxury car errors more heavily.

The AutoML approach beat the best manually-selected model — not because it generated more features, but because it found **CatBoost**, an algorithm that was never considered in the manual phase. This validates the core argument for AutoML: human model selection carries blind spots.

### Is $2,515 RMSE good enough?

Our primary measure is RMSE — it reflects the model's worst-case behaviour by penalising large errors more. An RMSE of $2,515 means the model's errors, when weighted towards the largest mistakes, are within roughly $2,515. For context, the average car in the dataset is priced around $7,400. For mid-range cars ($5,000–$10,000), typical predictions are within 10–20% of actual — commercially useful for price guidance. The secondary metric, MAE of $1,162, tells us the average dollar error across all predictions.

The model is weakest at the extremes:
- **Very cheap cars (below $500):** The model extrapolates poorly — too few training examples at this price level
- **Very expensive cars (above $50,000):** Luxury cars are under-represented in the training data, causing the model to underestimate prices by 30–40% on rare high-end listings

---

## Why We Chose RMSE as Our Primary Metric

We measured success using **RMSE (Root Mean Squared Error)** rather than the more common MAE (Mean Absolute Error) or R².

**The intuition:** RMSE penalises large errors more than small ones. A $10,000 mistake on a $50,000 luxury car is not "ten times worse" than a $1,000 mistake — in practice, it is catastrophically worse, because the customer would reject the valuation entirely. RMSE captures this asymmetry by squaring errors before averaging.

**Why not R²?** R² measures what proportion of price variance the model explains. On our dataset, a handful of very expensive cars dominate the variance — so a model can score a high R² even while making large dollar errors on everyday mid-range cars. R² is not a reliable signal for this problem.

---

## What the Model Learned

Using SHAP (a mathematical technique for explaining AI decisions), we identified the most important factors in car pricing:

1. **Car age** — the single largest driver of depreciation
2. **Mileage** — higher annual mileage = faster depreciation, independent of age
3. **Brand** — premium brands (Porsche, Lexus, BMW) command $7,000–$10,000+ price premiums
4. **Vehicle segment** — SUVs and executive cars (segments E, F, S) consistently price higher than economy hatchbacks
5. **Engine size** — larger displacement correlates with premium positioning
6. **Fuel type** — electric vehicles command a ~$21,000 premium over comparable petrol cars
7. **Transmission** — automatic transmission adds price premium across most segments
8. **Condition** — physical damage reduces value by ~$2,000 on average

These findings align with conventional car pricing intuition — which builds confidence that the model is learning real patterns rather than statistical noise.

---

## Transparency and Trust

Every prediction can be explained. We applied two complementary interpretability tools:

- **LIME:** For any individual car, LIME shows exactly which features pushed the price prediction up or down and by how much. This allows a user to understand *why* a specific car was priced the way it was.
- **SHAP:** Across all 11,227 test cars, SHAP quantifies the average contribution of each feature to predictions. This gives a global understanding of what the model considers important.

Model transparency is essential for building user trust — especially in a pricing context where customers may challenge or dispute automated valuations.

---

## Key Decisions Made (and Why)

| Decision | Rationale |
|---|---|
| **No log-transform of price** | Log-transform would make errors percentage-based, not dollar-based — less actionable for customers |
| **Stratified train/test split** | Ensures luxury cars are equally represented in training and test — prevents accidentally easy or hard test sets |
| **No preprocessing before the split** | All imputation, scaling, and encoding fitted on training data only — prevents data from the test set leaking into model training |
| **R² excluded from model selection** | R² is misleading on skewed distributions — see above |
| **CatBoost not manually tried in Phase 1** | Demonstrates the value of AutoML — human model selection missed the winning algorithm |

---

## Summary

| | Value |
|---|---|
| **Dataset** | 56,133 used car listings |
| **Best model** | PyCaret CatBoost (AutoML) |
| **Primary accuracy metric (RMSE)** | $2,515 |
| **Secondary metric (MAE)** | $1,162 |
| **Improvement over simple linear model** | 52% lower RMSE |
| **Improvement over best manual model** | $36 lower RMSE (1.4%) |
| **Platform** | Databricks (pipeline) + Google Colab (AutoML) |
| **Models benchmarked** | 20 (via PyCaret AutoML) |
| **Features used** | 100 (via Featuretools automated feature engineering) |
| **Model explainability** | Full — SHAP global + local, LIME individual predictions |
