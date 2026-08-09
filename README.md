# Used Car Price Prediction

Predicting used car selling prices from listing details, using Random Forest, XGBoost, LightGBM, and CatBoost — the two strongest performers get tuned via RandomizedSearchCV — served through a FastAPI endpoint.

## Overview

- **Target:** `selling_price`, log-transformed (`log1p`) to handle right-skew, reversed (`expm1`) for reporting
- **Data:** [Car details v3 dataset](https://www.kaggle.com/datasets/sajaabdalaal/car-details-v3csv) — used car listings with specs, mileage, ownership history, and selling price
- **Models:** Random Forest, XGBoost, LightGBM, CatBoost — all 4 compared untuned via 10-fold CV, then whichever *two* scored highest get tuned via `RandomizedSearchCV`, and whichever of those two wins becomes the saved model. Not hardcoded to CatBoost/LightGBM — the winners are decided from the actual CV scores each run, so RF or XGB could in principle be the ones tuned and saved.
- **Served via:** FastAPI, with bounds validation, one-hot consistency checks, API key auth, rate limiting, and error handling

## Pipeline

1. Drop `name` (high-cardinality free text) and `torque` (inconsistent unit formatting)
2. Sanity filters: remove implausible rows (seats > 10, km_driven > 500,000, year < 2000)
3. Derive `Car_age` from year (computed from the current year, not hardcoded, so this doesn't silently drift wrong in future years)
4. Strip units from `mileage`/`engine`/`max_power` (stored as strings like `"23.4 kmpl"`)
5. Split the row index *before* computing any derived statistics, so nothing below leaks test-row information into training
6. Impute missing `mileage`/`engine`/`max_power` (mean) and `seats` (mode) — all fit on the training split only
7. Remove outliers via IQR bounds on `selling_price`, computed from the training split only
8. Encode `transmission`/`owner` as ordinal, one-hot encode `fuel`/`seller_type`
9. Compare all 4 models untuned via 10-fold CV, then tune whichever two scored highest (decided from the actual CV means, not assumed ahead of time) via `RandomizedSearchCV`
10. Evaluate the winning model on a held-out test set, reporting R² in log-space and RMSE/MAE in actual currency units
11. Extract feature importance, branching on which model actually won — `CatBoostRegressor.get_feature_importance(Pool)` is CatBoost-only and would `AttributeError` for any of the other three models, so this is handled with an if/else (CatBoost vs. everyone else's `.feature_importances_`) rather than assuming CatBoost always wins
12. Save the model bundle (model, fill values, feature columns, target transform, training year, winning model name) to `car_price_{winner}_model.pkl` — the filename itself reflects which model won, rather than being hardcoded to one

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
- `winner` (which model — `"rf"`, `"xgb"`, `"catboost"`, or `"lgbm"` — is actually saved, useful since any of the four can win)
- `training_year` (the year `Car_age` was computed relative to during training — the API reads this instead of recomputing "now" at serving time, so its guidance doesn't silently drift once the model has been running a while without a retrain)

The API (`main.py`) includes:
- **Dynamic model loading** — finds whatever `car_price_*_model.pkl` the most recent training run produced (or reads a specific path from `CAR_PRICE_MODEL_PATH`), rather than assuming CatBoost always wins. Both the health-check and prediction responses report `model_used` so callers can see which model actually served the request.
- **Optional-field imputation** — `mileage`, `engine`, `max_power`, and `seats` can be omitted from a request; if so, they're filled in using the same train-only `fill_values`/`seats_mode` the training pipeline persisted, rather than rejecting incomplete input outright
- **Field-level bounds validation** (e.g. `km_driven` 0–500,000, `mileage` 0–50 kmpl) matching the training data's realistic ranges
- **One-hot consistency checks** — rejects requests where more than one `fuel_*` or more than one `seller_type_*` flag is set to `1`. All-zero on a group is valid (it means the dropped baseline category from training's `drop_first=True`: CNG for fuel, Dealer for seller_type); more than one flag set is not something the model was ever trained to expect
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
   python pipeline.py
   ```
   This writes `car_price_{winner}_model.pkl` (e.g. `car_price_catboost_model.pkl`, `car_price_lgbm_model.pkl`, `car_price_rf_model.pkl`, or `car_price_xgb_model.pkl`, depending on which model wins that run)
4. Create a `.env` file in this folder with your own API key:
   ```
   CAR_PRICE_API_KEY=your-own-secret-here
   ```
   Optionally, also set `CAR_PRICE_MODEL_PATH` if you want to pin the API to a specific saved bundle rather than auto-detecting the most recently trained one
5. Run the API:
   ```
   uvicorn main:app --reload
   ```
6. Test it interactively at `http://127.0.0.1:8000/docs`

## Possible Extensions

- A formal significance check (e.g. bootstrap CI) on the CatBoost-vs-LightGBM comparison, similar to the LR-vs-RF check in the heart disease project — currently the winner is decided by CV mean alone (with std reported alongside it, but no formal test)
- Wrap preprocessing + model into a single `ColumnTransformer`/`Pipeline` object, so encoding logic can't drift out of sync between training and serving
- Address the mileage unit-conflation issue described above
- A saved test suite (`test_main.py`) mirroring the heart disease project's coverage: known input → stable prediction, out-of-bounds rejected, missing optional field imputed correctly, one-hot consistency rejected, auth checks

## Files

- `pipeline.py` — full training pipeline: data cleaning, imputation, outlier removal, encoding, model comparison, tuning, evaluation, model saving
- `main.py` — FastAPI serving layer
- `Car details v3.csv` — input data, not included in repo. Download from the Kaggle link above and place it in this folder before running `pipeline.py`
- `car_price_{winner}_model.pkl` — saved model bundle, not included in repo (generated by running `pipeline.py`; filename depends on which model won that training run)
