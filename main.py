
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "calibrated_xgb_model.pkl"
THRESHOLD_PATH = BASE_DIR / "best_threshold.pkl"
STATIC_DIR = BASE_DIR / "static"


# ============================================================
# Load model and threshold once at application startup
# ============================================================

ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Loading credit risk model")

        model = joblib.load(MODEL_PATH)
        threshold = float(joblib.load(THRESHOLD_PATH))

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1")

        if not callable(getattr(model, "predict_proba", None)):
            raise TypeError("Model does not support predict_proba")

        ml_model["model"] = model
        ml_model["threshold"] = threshold

        logger.info("Model and threshold loaded successfully")

    except Exception:
        logger.exception("Application startup failed")
        raise

    try:
        yield
    finally:
        ml_model.clear()
        logger.info("Model resources cleared")


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Loan Prediction API",
    description="Credit risk prediction using calibrated XGBoost",
    version="1.1.0",
    lifespan=lifespan,
)


# Configure FRONTEND_ORIGIN in Render environment variables.
# Example: https://your-frontend.example.com
frontend_origin = os.getenv("FRONTEND_ORIGIN")

allowed_origins = (
    [origin.strip() for origin in frontend_origin.split(",") if origin.strip()]
    if frontend_origin
    else []
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ============================================================
# Request schema
# ============================================================

class LoanApplication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    person_age: int = Field(ge=18, le=100)
    person_income: int = Field(gt=0)
    person_home_ownership: str = Field(min_length=1)
    person_emp_length: float = Field(ge=0, le=60)
    loan_intent: str = Field(min_length=1)
    loan_grade: str = Field(min_length=1)
    loan_amnt: int = Field(gt=0)
    loan_int_rate: float = Field(gt=0, le=100)
    loan_percent_income: float = Field(ge=0, le=1)
    cb_person_default_on_file: str = Field(min_length=1)
    cb_person_cred_hist_length: int = Field(ge=0)


# ============================================================
# Health checks
# ============================================================

@app.get("/health")
def health_check():
    model_loaded = ml_model.get("model") is not None
    threshold_loaded = ml_model.get("threshold") is not None

    return {
        "status": "healthy" if model_loaded and threshold_loaded else "not_ready",
        "model_loaded": model_loaded,
        "threshold_loaded": threshold_loaded,
    }


@app.get("/ready")
def readiness_check():
    if (
        ml_model.get("model") is None
        or ml_model.get("threshold") is None
    ):
        raise HTTPException(
            status_code=503,
            detail="Model resources are not ready",
        )

    return {"status": "ready"}


# ============================================================
# Prediction endpoint
# ============================================================

@app.post("/predict")
def predict_loan_default(data: LoanApplication):
    try:
        model = ml_model.get("model")
        threshold = ml_model.get("threshold")

        if model is None or threshold is None:
            raise HTTPException(
                status_code=503,
                detail="Prediction service is not ready",
            )

        input_df = pd.DataFrame([data.model_dump()])

        probabilities = model.predict_proba(input_df)

        if probabilities.shape[1] != 2:
            raise ValueError("Expected a binary classification model")

        probability = float(probabilities[0, 1])
        predicted_class = int(probability >= threshold)

        if not 0.0 <= probability <= 1.0:
            raise ValueError("Model returned an invalid probability")

        risk = "High Risk" if predicted_class == 1 else "Low Risk"
        decision = (
            "Likely to Default"
            if predicted_class == 1
            else "Likely to Repay"
        )

        # Log the outcome, not the applicant's personal data.
        logger.info("Prediction completed successfully")

        return {
            "status": "success",
            "default_probability": round(probability, 4),
            "default_probability_percentage": round(
                probability * 100, 2
            ),
            "default_prediction": predicted_class,
            "threshold": round(threshold, 4),
            "risk": risk,
            "decision": decision,
            "model": MODEL_PATH.name,
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check server logs.",
        )


# ============================================================
# Serve frontend files
# Keep this mount after the API route definitions.
# ============================================================

if STATIC_DIR.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=str(STATIC_DIR), html=True),
        name="static",
    )
else:
    logger.warning("Static directory not found: %s", STATIC_DIR)
