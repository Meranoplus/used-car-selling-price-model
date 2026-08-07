import pandas as pd 
import numpy as np
from catboost import Pool
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, cross_val_score
import joblib


# ── Config ────────────────────────────────────────────────

config = {
    'random_state': 42,
    'test_size':    0.2,
    'n_splits':     10,
    'current_year': pd.Timestamp.now().year,   
}

# ── model params (untuned, used for the initial CV comparison below) ──

rf_params = {
    'n_estimators': 300,
    'random_state': config["random_state"],
    'n_jobs':       -1,
}

xgb_params = {
    'n_estimators':  500,
    'learning_rate': 0.1,
    'max_depth':     6,
    'random_state':  config["random_state"],
    'n_jobs':        -1,
    'objective':     'reg:squarederror',
}

lgbm_params = {
    'n_estimators':  500,
    'learning_rate': 0.05,
    'num_leaves':    31,
    'random_state':  config["random_state"],
    'n_jobs':        -1,
    'verbose':       -1,
    'objective':     'regression',
}

cat_params = {
    'iterations':    500,
    'learning_rate': 0.1,
    'depth':         6,
    'random_state':  config["random_state"],
    'verbose':       False,
    'loss_function': 'RMSE',
    'allow_writing_files': False,
}

# RandomizedSearchCV grids — only CatBoost and LightGBM get tuned below,
# since they came out ahead in the initial untuned CV comparison.
cat_grid = {
    'iterations':    [300, 500, 700],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth':         [4, 6, 8, 10],
    'l2_leaf_reg':   [1, 3, 5, 7],
}

lgbm_grid = {
    'n_estimators':  [300, 500, 700],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves':    [31, 63, 127],
    'max_depth':     [-1, 6, 8, 10],
    'min_child_samples': [10, 20, 30],
}

df = pd.read_csv("Car details v3.csv")

# Drop columns with no reusable predictive value:
# - name: free-text, extremely high cardinality, mostly redundant with other specs
# - torque: inconsistent unit formatting across rows, not worth the cleanup here
df = df.drop(columns=["name", "torque"])
df = df.drop_duplicates()

# Basic sanity filters — remove implausible/likely-erroneous rows
# (e.g. a listing with 20 seats, or a car "driven" more than the odometer
# would realistically show for a used-car listing) before any modeling.
df = df[df['seats'] <= 10]
df = df[df['km_driven'] <= 500000]
df = df[df['year'] >= 2000]

# Age is more directly useful to a model than a raw model year
# (avoids the model having to learn "smaller year = older car").
df['Car_age'] = config["current_year"] - df['year']
df = df.drop(columns=['year'])

# strip units — source columns store values as strings like "23.4 kmpl" /
# "1248 CC" / "74 bhp"; extract just the numeric portion for modeling.
#
# KNOWN LIMITATION: mileage is reported in kmpl for petrol/diesel cars but
# km/kg for CNG/LPG cars — two different physical units sharing one column.
# This line treats them as one continuous numeric feature, silently
# conflating a CNG car's mileage value with a diesel car's. Not fixed here
# — would need normalizing by fuel type or excluding CNG/LPG rows from this
# column, similar to how torque was dropped. Worth deciding on deliberately
# before trusting mileage-related feature importance too literally.
df['mileage'] = df['mileage'].str.extract(r'(\d+\.?\d*)').astype(float)
df['engine']  = df['engine'].str.extract(r'(\d+\.?\d*)').astype(float)
df['max_power'] = df['max_power'].str.extract(r'(\d+\.?\d*)').astype(float)

# Split the INDEX now, before computing anything derived from the data
# (fill values, outlier bounds, etc.), so none of those computations leak
# information from the test rows into training.
train_idx, test_idx = train_test_split(
    df.index, random_state=config["random_state"], test_size=config["test_size"]
)

# Save fill values from TRAIN ROWS ONLY — these get persisted alongside
# the final model (see joblib.dump below) so a live API can impute
# missing input the same way, using train-only statistics.
fill_values = {
    'mileage':   df.loc[train_idx, 'mileage'].mean(),
    'engine':    df.loc[train_idx, 'engine'].mean(),
    'max_power': df.loc[train_idx, 'max_power'].mean(),
}
seats_mode = df.loc[train_idx, 'seats'].mode()[0]

# Fill as normal — applied to the whole df, but the fill values themselves
# came from train only, so no leakage.
df["mileage"]   = df["mileage"].fillna(fill_values['mileage'])
df["engine"]    = df["engine"].fillna(fill_values['engine'])
df["max_power"] = df["max_power"].fillna(fill_values['max_power'])
df["seats"]     = df["seats"].fillna(seats_mode)

# Outlier removal via IQR, bounds computed from TRAIN ROWS ONLY — extreme
# selling_price values (luxury cars, obvious data-entry errors) can otherwise
# dominate the loss function for tree/linear regressors alike.
Q1 = df.loc[train_idx, 'selling_price'].quantile(0.25)
Q3 = df.loc[train_idx, 'selling_price'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df['selling_price'] >= lower) & (df['selling_price'] <= upper)]
# Some rows just got dropped by the outlier filter above — refresh the
# saved index lists so train_idx/test_idx don't reference removed rows.
train_idx = train_idx.intersection(df.index)
test_idx  = test_idx.intersection(df.index)

# Encode categoricals: ordinal mappings for transmission/owner (there's a
# natural order/binary meaning to preserve), one-hot for fuel/seller_type
# (no inherent order between categories).
df['transmission'] = df["transmission"].map({"Manual":0, "Automatic":1})
df["owner"] = df["owner"].map({"First Owner":1, "Second Owner":2, "Third Owner":3, "Fourth & Above Owner":4, "Test Drive Car":0})
df = pd.get_dummies(df, columns=["fuel", "seller_type"], drop_first=True)

X = df.drop(columns=["selling_price"]).copy()
# Log-transform the target — selling_price is heavily right-skewed
# (a handful of expensive cars would otherwise dominate the loss);
# predictions get reversed with np.expm1() before being reported.
y = np.log1p(df["selling_price"])

# Use the index split saved earlier, not a fresh train_test_split — keeps
# every downstream step (fill values, outlier bounds, this split) consistent
# with the exact same train/test row assignment.
X_train, X_test = X.loc[train_idx], X.loc[test_idx]
y_train, y_test = y.loc[train_idx], y.loc[test_idx]

models = {
    "rf": RandomForestRegressor(**rf_params),
    "catboost": CatBoostRegressor(**cat_params),
    "xgb": XGBRegressor(**xgb_params),
    "lgbm": LGBMRegressor(**lgbm_params)
}

kfold = KFold(n_splits=config["n_splits"], random_state=config["random_state"], shuffle=True)

# Initial untuned comparison across candidate models — decides which
# ones are worth the extra time/cost of hyperparameter search below.
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring="r2")

    print(f"{name}: R2 = {scores.mean():.4f} (+/- {scores.std():.4f})")

# Only CatBoost and LightGBM get tuned — RandomizedSearchCV (not
# GridSearchCV) is used here since each grid has hundreds of combinations,
# too many to search exhaustively; n_iter=20 samples a fixed subset instead.
catboost_rscv = RandomizedSearchCV(
    estimator=CatBoostRegressor(verbose=False, allow_writing_files=False, random_state=config["random_state"]),
    param_distributions=cat_grid,
    n_iter=20, cv=kfold,
    scoring="r2",
    n_jobs=-1, refit=True, verbose=0,
    random_state=config["random_state"]
)

lgbm_rscv = RandomizedSearchCV(
    estimator=LGBMRegressor(verbose=-1, random_state=config["random_state"]),
    param_distributions=lgbm_grid,
    n_iter=20, cv=kfold,
    scoring="r2",
    n_jobs=-1, refit=True, verbose=0,
    random_state=config["random_state"]
)

catboost_rscv.fit(X_train, y_train)
print(f"  CatBoost best r2 (CV): {catboost_rscv.best_score_:.4f}")

lgbm_rscv.fit(X_train, y_train)
print(f"  lgbm best r2 (CV): {lgbm_rscv.best_score_:.4f}")

# Report each model's CV standard deviation alongside its score — a higher
# mean with much higher variance isn't automatically the better pick.
cat_std  = catboost_rscv.cv_results_['std_test_score'][catboost_rscv.best_index_]
lgbm_std = lgbm_rscv.cv_results_['std_test_score'][lgbm_rscv.best_index_]
print(f"\nCatBoost: {catboost_rscv.best_score_:.4f} ± {cat_std:.4f}")
print(f"LGBM:     {lgbm_rscv.best_score_:.4f} ± {lgbm_std:.4f}")


if catboost_rscv.best_score_  >= lgbm_rscv.best_score_:
    winner, final_model = "catboost", catboost_rscv.best_estimator_

else: 
    winner, final_model = "lgbm", lgbm_rscv.best_estimator_
print(f"\nWinner: {winner}")

# Final holdout evaluation — first time X_test/y_test are touched at all,
# confirms the winning model's CV score generalizes to unseen data.
final_model.fit(X_train, y_train)
y_pred        = final_model.predict(X_test)
# Reverse the log1p transform to report errors in actual currency units,
# not log-price units (which wouldn't mean anything to a reader).
y_pred_actual = np.expm1(y_pred)
y_test_actual = np.expm1(y_test)

print("\nFINAL TEST RESULTS")
print("=" * 50)
print(f"R²:   {r2_score(y_test, y_pred):.4f}   (log-price space — the space the model was optimized in)")
print(f"RMSE: {root_mean_squared_error(y_test_actual, y_pred_actual):,.0f}   (actual currency units, post-expm1)")
print(f"MAE:  {mean_absolute_error(y_test_actual, y_pred_actual):,.0f}   (actual currency units, post-expm1)")

# Leakage sanity check — no single feature should dominate; a smooth,
# spread-out correlation table is consistent with a genuine multi-factor
# pricing model rather than a hidden shortcut in the data.
print("\ncorr:")
print(df.corr(numeric_only=True)["selling_price"].sort_values(ascending=False).head(20))

print("\nfeature importance")
# get_feature_importance(Pool) is CatBoost-only — branch on which model
# actually won, since lgbm can win this comparison too (LGBMRegressor
# uses .feature_importances_ instead, no Pool argument).
if winner == "catboost":
    train_pool = Pool(X_train, y_train)
    importance = pd.Series(final_model.get_feature_importance(train_pool), index=X_train.columns)
else:
    importance = pd.Series(final_model.feature_importances_, index=X_train.columns)
print(importance.sort_values(ascending=False).head(15))

# Persist everything a live API needs to reproduce this exact pipeline at
# inference time: the model itself, plus every train-only statistic used
# for imputation, plus the exact column order the model expects, plus
# metadata so a loader doesn't need tribal knowledge of this script.
out_path = f"car_price_{winner}_model.pkl"
joblib.dump({
    "model":             final_model,
    "winner":            winner,               # "catboost" or "lgbm" — which model this actually is
    "fill_values":       fill_values,
    "seats_mode":        seats_mode,
    "feature_columns":   list(X_train.columns),
    "target_transform":  "log1p",               # predictions need np.expm1() applied — self-documenting now
}, out_path)
print("Model saved!")
