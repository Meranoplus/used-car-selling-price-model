import glob
import logging
import os
import secrets

import numpy as np
import pandas as pd
import joblib

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from pydantic import BaseModel, Field, model_validator
from typing import Optional

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("car_price_api")

# ── Rate limiting ────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Load trained artifacts ─────────────────────────────────
# The training script now saves to f"car_price_{winner}_model.pkl", so the
# winning model type isn't fixed ahead of time — glob for whatever the most
# recent training run actually produced instead of hardcoding "catboost".
MODEL_PATH = os.environ.get("CAR_PRICE_MODEL_PATH")
if not MODEL_PATH:
    candidates = sorted(
        glob.glob("car_price_*_model.pkl"),
        key=os.path.getmtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(
            "No car_price_*_model.pkl found and CAR_PRICE_MODEL_PATH not set — "
            "run the training script first, or point CAR_PRICE_MODEL_PATH at a bundle."
        )
    MODEL_PATH = candidates[0]

bundle          = joblib.load(MODEL_PATH)
model           = bundle["model"]
winner          = bundle.get("winner", "unknown")
feature_columns = bundle["feature_columns"]
fill_values     = bundle["fill_values"]      # mileage/engine/max_power means, train-only
seats_mode      = bundle["seats_mode"]
target_transform = bundle.get("target_transform", "log1p")

logger.info("Loaded %s model (%s) from %s", winner, target_transform, MODEL_PATH)

# training script computes Car_age as config["current_year"] - year at whatever
# moment the model was actually trained, and persists that year in the bundle
# (see "training_year" in the joblib.dump). Read it from there rather than
# recomputing pd.Timestamp.now().year here — "now" is when the API *started*,
# not when the model was *trained*, and those drift apart the longer a model
# runs without being retrained. Falls back to "now" with a warning only for
# bundles saved before this fix, which won't have the key.
if "training_year" in bundle:
    MODEL_TRAINING_YEAR = bundle["training_year"]
else:
    MODEL_TRAINING_YEAR = pd.Timestamp.now().year
    logger.warning(
        "Bundle at %s has no 'training_year' (trained before this fix) — "
        "falling back to the current year (%d), which may not be when this "
        "model was actually trained. Retrain to embed the real year.",
        MODEL_PATH, MODEL_TRAINING_YEAR,
    )

# ── API key auth ────────────────────────────────────────────
# Fails at startup rather than falling back to a guessable default.
API_KEY = os.environ.get("CAR_PRICE_API_KEY")
if not API_KEY:
    raise RuntimeError("CAR_PRICE_API_KEY must be set — refusing to start with no auth configured.")

def verify_api_key(x_api_key: str = Header(...)):
    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


class CarFeatures(BaseModel):
    km_driven:                    int             = Field(..., ge=0, le=500_000, description="Odometer reading in km")
    transmission:                 int             = Field(..., ge=0, le=1, description="0=Manual, 1=Automatic")
    owner:                        int             = Field(..., ge=0, le=4, description="0=Test Drive, 1=First, 2=Second, 3=Third, 4=Fourth+")
    # mileage/engine/max_power/seats are optional — if omitted, they're
    # imputed with the same train-only statistics the training script
    # persisted (fill_values / seats_mode), matching how missing values
    # were handled during training.
    mileage:                      Optional[float] = Field(None, gt=0, le=50, description="Fuel efficiency (kmpl). Imputed from training-set mean if omitted.")
    engine:                       Optional[float] = Field(None, gt=0, le=6_000, description="Engine displacement in CC. Imputed from training-set mean if omitted.")
    max_power:                    Optional[float] = Field(None, gt=0, le=500, description="Max power in bhp. Imputed from training-set mean if omitted.")
    seats:                        Optional[int]   = Field(None, ge=2, le=10, description="Number of seats. Imputed from training-set mode if omitted.")
    Car_age:                      int             = Field(..., ge=0, le=30, description=f"Approx. (model training year) - manufacturing year. Model was last trained in {MODEL_TRAINING_YEAR}.")
    fuel_Diesel:                  int             = Field(..., ge=0, le=1, description="1 if Diesel, else 0")
    fuel_LPG:                     int             = Field(..., ge=0, le=1, description="1 if LPG, else 0")
    fuel_Petrol:                  int             = Field(..., ge=0, le=1, description="1 if Petrol, else 0")
    seller_type_Individual:       int             = Field(..., ge=0, le=1, description="1 if Individual seller, else 0")
    seller_type_Trustmark_Dealer: int             = Field(..., ge=0, le=1, alias="seller_type_Trustmark Dealer", description="1 if Trustmark Dealer, else 0")

    class Config:
        populate_by_name = True

    @model_validator(mode="after")
    def _check_one_hot_groups(self):
        # fuel_* and seller_type_* are one-hot columns from the training
        # pipeline's pd.get_dummies(..., drop_first=True) — at most one flag
        # per group may be 1. All-zero is valid (it means the dropped
        # baseline category: CNG for fuel, Dealer for seller_type), but more
        # than one flag set is a nonsensical input the model was never
        # trained to expect.
        fuel_flags = [self.fuel_Diesel, self.fuel_LPG, self.fuel_Petrol]
        if sum(fuel_flags) > 1:
            raise ValueError(
                "At most one of fuel_Diesel, fuel_LPG, fuel_Petrol may be 1 "
                "(all-zero means CNG, the dropped baseline category)."
            )

        seller_flags = [self.seller_type_Individual, self.seller_type_Trustmark_Dealer]
        if sum(seller_flags) > 1:
            raise ValueError(
                "At most one of seller_type_Individual, seller_type_Trustmark_Dealer "
                "may be 1 (all-zero means Dealer, the dropped baseline category)."
            )
        return self


class PredictionResponse(BaseModel):
    predicted_price: float
    currency:        str
    model_used:      str


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("10/minute")
def predict(request: Request, car: CarFeatures, _auth: None = Depends(verify_api_key)):
    # by_alias=True already renders the aliased key with the space
    # ("seller_type_Trustmark Dealer") to match the training column name —
    # no manual rename/pop needed on top of it.
    input_dict = car.model_dump(by_alias=True)

    # Impute any omitted optional fields using the same train-only
    # statistics persisted by the training script.
    if input_dict["mileage"] is None:
        input_dict["mileage"] = fill_values["mileage"]
    if input_dict["engine"] is None:
        input_dict["engine"] = fill_values["engine"]
    if input_dict["max_power"] is None:
        input_dict["max_power"] = fill_values["max_power"]
    if input_dict["seats"] is None:
        input_dict["seats"] = seats_mode

    try:
        features = pd.DataFrame([input_dict])
        features = features[feature_columns]   # enforce exact column order used in training

        prediction = model.predict(features)[0]
        # reverse whatever transform was applied to selling_price in training
        price = float(np.expm1(prediction)) if target_transform == "log1p" else float(prediction)
    except Exception:
        logger.exception("Inference failed for input: %s", input_dict)
        raise HTTPException(status_code=500, detail="Prediction failed — please check your input.")

    logger.info(
        "predict: km_driven=%s Car_age=%s -> predicted_price=%.2f",
        input_dict["km_driven"], input_dict["Car_age"], price
    )

    return PredictionResponse(
        predicted_price=round(price, 2),
        currency="INR",
        model_used=winner,
    )


@app.get("/")
def root():
    return {"status": "ok", "message": "Car Price Predictor API is running", "model_used": winner}
