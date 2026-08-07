# Used Car Price Prediction

Predicting used car selling prices from listing details, using CatBoost/LightGBM (tuned via RandomizedSearchCV) benchmarked against Random Forest and XGBoost — served through a FastAPI endpoint.

## Overview

- **Target:** `selling_price`, log-transformed (`log1p`) to handle right-skew, reversed (`expm1`) for reporting
- **Data:** [Car details v3 dataset](https://www.kaggle.com/datasets/sajaabdalaal/car-details-v3csv) — used car listings with specs, mileage, ownership history, and selling price
- **Models:** Random Forest, XGBoost, LightGBM, CatBoost — CatBoost/LightGBM tuned via `RandomizedSearchCV`, whichever wins becomes the saved model
- **Served via:** FastAPI, with bounds validation, API key auth, rate limiting, and error handling

## Pipeline

1. Drop `name` (high-cardinality free text) and `torque` (inconsistent unit formatting)
2. Sanity filters: remove implausible rows (seats > 10, km_driven > 500,000, year < 2000)
3. Derive `Car_age` from year (computed from the current year, not hardcoded, so this doesn't silently drift wrong in future years)
4. Strip units from `mileage`/`engine`/`max_power` (stored as strings like `"23.4 kmpl"`)
5. Split the row index *before* computing any derived statistics, so nothing below leaks test-row information into training
6. Impute missing `mileage`/`engine`/`max_power` (mean) and `seats` (mode) — all fit on the training split only
7. Remove outliers via IQR bounds on `selling_price`, computed from the training split only
8. Encode `transmission`/`owner` as ordinal, one-hot encode `fuel`/`seller_type`
9. Compare all 5 models untuned via 10-fold CV, then tune CatBoost and LightGBM (the two strongest performers) via `RandomizedSearchCV`
10. Evaluate the winning model on a held-out test set, reporting R² in log-space and RMSE/MAE in actual currency units
11. Extract feature importance, branching on which model actually won — `CatBoostRegressor.get_feature_importance(Pool)` is CatBoost-only and would `AttributeError` if `LGBMRegressor` won instead, so this is handled with an if/else rather than assuming CatBoost always wins
12. Save the model bundle (model, fill values, feature columns, target transform, winning model name) for serving

## A Known Limitation, Left Deliberately Unfixed

`mileage` is reported in **kmpl** for petrol/diesel cars but **km/kg** for CNG/LPG cars — two different physical units sharing one column. This pipeline strips the unit string and treats the number as one continuous feature, which silently conflates a CNG car's mileage with a diesel car's. This wasn't fixed here (it would require normalizing by fuel type or excluding CNG/LPG rows, similar to how `torque` was dropped) — flagged as a known limitation rather than silently ignored. Worth keeping in mind if `mileage`'s feature importance looks larger or smaller than expected.

This same conflation propagates into the API's validation layer: `main.py` bounds `mileage` to `0–50`, labeled `kmpl` in the field description. Typical CNG mileage (roughly 20s–30s km/kg for many vehicles) falls *inside* that range, so the validator doesn't actually catch a km/kg value submitted for a CNG/LPG car — it silently accepts it as if it were a kmpl reading. The bound only meaningfully protects against implausible values within the petrol/diesel unit system; it isn't a real safeguard against the unit mismatch itself.

## Serving

The saved model bundle includes everything the API needs to reproduce the training pipeline's preprocessing at inference time:
- The winning model itself
- `fill_values` (training-set means for `mileage`/`engine`/`max_power`)
- `seats_mode`
- `feature_columns` (exact column order the model expects)
- `target_transform` (`"log1p"`, so predictions are known to need `expm1()` reversal — not tribal knowledge)
- `winner` (which model — `"catboost"` or `"lgbm"` — is actually saved, useful since either can win)

The API (`main.py`) includes:
- Field-level bounds validation (e.g. `km_driven` 0–500,000, `mileage` 0–50 kmpl) matching the training data's realistic ranges
- API key authentication (fails at startup if no key is configured, rather than falling back to a guessable default)
- Rate limiting (10 requests/minute per IP)
- Error handling around inference (returns a clean `500` without leaking internals on unexpected failures)
- Logging of each prediction

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Download the dataset from the Kaggle link above and place `Car details v3.csv` in this folder
3. Run the training pipeline to generate the saved model:
   ```
   python car_price_pipeline.py
   ```
4. Create a `.env` file in this folder with your own API key:
   ```
   CAR_PRICE_API_KEY=your-own-secret-here
   ```
5. Run the API:
   ```
   uvicorn main:app --reload
   ```
6. Test it interactively at `http://127.0.0.1:8000/docs`

## Possible Extensions

- A formal significance check (e.g. bootstrap CI) on the CatBoost-vs-LightGBM comparison, similar to the LR-vs-RF check in the heart disease project — currently the winner is decided by CV mean alone (with std reported alongside it, but no formal test)
- Wrap preprocessing + model into a single `ColumnTransformer`/`Pipeline` object, so encoding logic can't drift out of sync between training and serving
- Address the mileage unit-conflation issue described above
- A saved test suite (`test_main.py`) mirroring the heart disease project's coverage: known input → stable prediction, out-of-bounds rejected, missing optional field imputed correctly, auth checks

## Files

- `car_price_pipeline.py` — full training pipeline: data cleaning, imputation, outlier removal, encoding, model comparison, tuning, evaluation, model saving
- `main.py` — FastAPI serving layer
- `Car details v3.csv` — input data, not included in repo. Download from the Kaggle link above and place it in this folder before running `car_price_pipeline.py`
- `car_price_catboost_model.pkl` — saved model bundle, not included in repo (generated by running `car_price_pipeline.py`)
