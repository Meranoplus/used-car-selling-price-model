"""
feedback_tracking.py
Add this file to your project folder, next to main.py.

WHAT THIS DOES:
- Every time the API makes a prediction, it saves it (with an ID) to a small
  local database file called "predictions.db".
- Later, when you find out what the car ACTUALLY sold for, you call a new
  endpoint and give it that ID + the real price.
- If a car gets delisted WITHOUT selling, you can mark that too — this
  matters, because a car that never sold is information (probably it was
  overpriced), not just "missing data."
- A stats endpoint lets you check how accurate your model has been,
  windowed by time, and broken out by status (sold vs. still listed vs.
  delisted unsold).

HOW TO WIRE IT INTO main.py:

1. At the top of main.py, add:
     from feedback_tracking import init_db, log_prediction, log_outcome, get_accuracy_stats

2. Right after your FastAPI app is created, add:
     init_db()

3. Inside your existing /predict endpoint, right before you `return` the
   prediction, add:
     prediction_id = log_prediction(
         input_data.dict(),
         predicted_price,
         model_version="v1"  # bump this string whenever you retrain/redeploy
     )
   ...and include `"prediction_id": prediction_id` in the JSON you return.

4. Add these endpoints anywhere in main.py:

     from pydantic import BaseModel

     class OutcomeReport(BaseModel):
         prediction_id: int
         status: str          # "sold" or "delisted_unsold"
         actual_price: float | None = None  # required if status == "sold"

     @app.post("/report-outcome")
     def report_outcome(report: OutcomeReport):
         if report.status == "sold" and report.actual_price is None:
             raise HTTPException(status_code=400, detail="actual_price required when status is 'sold'")
         success = log_outcome(report.prediction_id, report.status, report.actual_price)
         if not success:
             raise HTTPException(status_code=404, detail="prediction_id not found")
         return {"message": "Outcome recorded."}

     @app.get("/model-stats")
     def model_stats(days: int = 30):
         return get_accuracy_stats(window_days=days)

   (days is optional — /model-stats defaults to a 30-day rolling window,
   or call /model-stats?days=90 etc.)

NOTE ON AUTH: these endpoints currently have no auth. Fine for local dev.
Before this is public-facing, put report-outcome and model-stats behind
whatever auth your app already uses (API key header, etc.) — otherwise
anyone can inject fake outcomes into your accuracy stats.
"""

import sqlite3
import json
from datetime import datetime, timezone

DB_PATH = "predictions.db"

VALID_STATUSES = {"sold", "delisted_unsold"}


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")  # lets reads/writes overlap a bit better under FastAPI
    return conn


def init_db():
    """Creates the predictions table if it doesn't exist yet. Safe to call every startup."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_data TEXT NOT NULL,
            predicted_price REAL NOT NULL,
            model_version TEXT NOT NULL,
            predicted_at TEXT NOT NULL,
            status TEXT,                 -- NULL = no outcome yet, 'sold', or 'delisted_unsold'
            actual_price REAL,           -- only set when status = 'sold'
            outcome_reported_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(input_data: dict, predicted_price: float, model_version: str = "unknown") -> int:
    """Call this right after the model makes a prediction. Returns the new row's ID."""
    conn = _get_conn()
    cursor = conn.execute(
        "INSERT INTO predictions (input_data, predicted_price, model_version, predicted_at) "
        "VALUES (?, ?, ?, ?)",
        (json.dumps(input_data), predicted_price, model_version, datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def log_outcome(prediction_id: int, status: str, actual_price: float | None = None) -> bool:
    """
    Call this once you know what happened to the car:
      - status="sold", actual_price=<real sale price>
      - status="delisted_unsold" (no actual_price needed)
    Returns False if the ID doesn't exist or status is invalid.
    """
    if status not in VALID_STATUSES:
        return False

    conn = _get_conn()
    cursor = conn.execute(
        "UPDATE predictions SET status = ?, actual_price = ?, outcome_reported_at = ? WHERE id = ?",
        (status, actual_price, datetime.now(timezone.utc).isoformat(), prediction_id)
    )
    conn.commit()
    updated = cursor.rowcount > 0
    conn.close()
    return updated


def get_accuracy_stats(window_days: int = 30) -> dict:
    """
    Looks at predictions made in the last `window_days` days and reports:
      - accuracy stats for the ones that sold (MAE, MAPE)
      - how many are still waiting on an outcome
      - how many were delisted without selling (a drift/pricing warning sign)
    """
    conn = _get_conn()
    cutoff = datetime.now(timezone.utc).timestamp() - (window_days * 86400)
    cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()

    rows = conn.execute(
        "SELECT predicted_price, actual_price, status FROM predictions WHERE predicted_at >= ?",
        (cutoff_iso,)
    ).fetchall()
    conn.close()

    if not rows:
        return {"message": "No predictions in this window.", "window_days": window_days, "count": 0}

    sold = [(p, a) for p, a, s in rows if s == "sold" and a is not None]
    delisted_unsold = sum(1 for _, _, s in rows if s == "delisted_unsold")
    still_pending = sum(1 for _, _, s in rows if s is None)

    result = {
        "window_days": window_days,
        "total_predictions": len(rows),
        "sold_count": len(sold),
        "delisted_unsold_count": delisted_unsold,
        "still_pending_count": still_pending,
    }

    if sold:
        errors = [abs(pred - actual) for pred, actual in sold]
        pct_errors = [abs(pred - actual) / actual * 100 for pred, actual in sold if actual != 0]
        result["mean_absolute_error"] = round(sum(errors) / len(errors), 2)
        result["mean_absolute_percentage_error"] = (
            round(sum(pct_errors) / len(pct_errors), 2) if pct_errors else None
        )
    else:
        result["mean_absolute_error"] = None
        result["mean_absolute_percentage_error"] = None

    # crude early-warning signal: rising unsold rate among *resolved* outcomes
    resolved = len(sold) + delisted_unsold
    if resolved > 0:
        result["unsold_rate_of_resolved"] = round(delisted_unsold / resolved, 3)

    return result