import logging
import os
import secrets

import numpy as np
import pandas as pd
import joblib

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from pydantic import BaseModel, Field

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
# NOTE: this must match the out_path your training script actually saved to.
bundle           = joblib.load("car_price_catboost_model.pkl")
model            = bundle["model"]
feature_columns  = bundle["feature_columns"]
fill_values      = bundle["fill_values"]      # mileage/engine/max_power means, train-only
seats_mode       = bundle["seats_mode"]

# ── API key auth ────────────────────────────────────────────
# Fails at startup rather than falling back to a guessable default.
API_KEY = os.environ.get("CAR_PRICE_API_KEY")
if not API_KEY:
    raise RuntimeError("CAR_PRICE_API_KEY must be set — refusing to start with no auth configured.")

def verify_api_key(x_api_key: str = Header(...)):
    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


class CarFeatures(BaseModel):
    km_driven:                    int   = Field(..., ge=0, le=500_000, description="Odometer reading in km")
    transmission:                 int   = Field(..., ge=0, le=1, description="0=Manual, 1=Automatic")
    owner:                        int   = Field(..., ge=0, le=4, description="0=Test Drive, 1=First, 2=Second, 3=Third, 4=Fourth+")
    mileage:                      float = Field(..., gt=0, le=50, description="Fuel efficiency (kmpl)")
    engine:                       float = Field(..., gt=0, le=6_000, description="Engine displacement in CC")
    max_power:                    float = Field(..., gt=0, le=500, description="Max power in bhp")
    seats:                        int   = Field(..., ge=2, le=10, description="Number of seats")
    Car_age:                      int   = Field(..., ge=0, le=30, description="2024 - manufacturing year")
    fuel_Diesel:                  int   = Field(..., ge=0, le=1, description="1 if Diesel, else 0")
    fuel_LPG:                     int   = Field(..., ge=0, le=1, description="1 if LPG, else 0")
    fuel_Petrol:                  int   = Field(..., ge=0, le=1, description="1 if Petrol, else 0")
    seller_type_Individual:       int   = Field(..., ge=0, le=1, description="1 if Individual seller, else 0")
    seller_type_Trustmark_Dealer: int   = Field(..., ge=0, le=1, alias="seller_type_Trustmark Dealer", description="1 if Trustmark Dealer, else 0")

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    predicted_price: float
    currency:        str


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("10/minute")
def predict(request: Request, car: CarFeatures, _auth: None = Depends(verify_api_key)):
    input_dict = car.model_dump(by_alias=True)
    # rename to match the training column name (contains a space, invalid as a Python identifier)
    input_dict["seller_type_Trustmark Dealer"] = input_dict.pop("seller_type_Trustmark_Dealer")

    try:
        features = pd.DataFrame([input_dict])
        features = features[feature_columns]   # enforce exact column order used in training

        log_prediction = model.predict(features)[0]
        price = float(np.expm1(log_prediction))   # reverse the log1p transform applied to selling_price in training
    except Exception:
        logger.exception("Inference failed for input: %s", input_dict)
        raise HTTPException(status_code=500, detail="Prediction failed — please check your input.")

    logger.info(
        "predict: km_driven=%s Car_age=%s -> predicted_price=%.2f",
        input_dict["km_driven"], input_dict["Car_age"], price
    )

    return PredictionResponse(
        predicted_price=round(price, 2),
        currency="INR"
    )


@app.get("/")
def root():
    return {"status": "ok", "message": "Car Price Predictor API is running"}