from contextlib import asynccontextmanager

from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ============================================================
# Global storage for model and threshold
# ============================================================

ml_model = {}


# ============================================================
# Load model when application starts
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    ml_model["model"] = joblib.load(
        "calibrated_xgb_model.pkl"
    )

    ml_model["threshold"] = joblib.load(
        "best_threshold.pkl"
    )

    print("Model and threshold loaded successfully!")

    yield

    # Clean up when application shuts down
    ml_model.clear()

    print("Model resources cleared.")


# ============================================================
# Create FastAPI application
# ============================================================

app = FastAPI(
    title="Loan Prediction API",
    description="Loan Default Prediction using Calibrated XGBoost",
    version="1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ============================================================
# Request schema
# ============================================================

class LoanApplication(BaseModel):

    person_age: int

    person_income: int

    person_home_ownership: str

    person_emp_length: float

    loan_intent: str

    loan_grade: str

    loan_amnt: int

    loan_int_rate: float

    loan_percent_income: float

    cb_person_default_on_file: str

    cb_person_cred_hist_length: int



# ============================================================
# Health check
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": ml_model.get("model") is not None,
        "threshold_loaded": ml_model.get("threshold") is not None
    }


# ============================================================
# Prediction route
# ============================================================

@app.post("/predict")
def predict_loan_default(data: LoanApplication):

    # --------------------------------------------------------
    # Convert Pydantic object into DataFrame
    # --------------------------------------------------------

    input_df = pd.DataFrame(
        [data.model_dump()]
    )

    # --------------------------------------------------------
    # Get probability from trained model
    # --------------------------------------------------------

    prediction_proba = (
        ml_model["model"]
        .predict_proba(input_df)[:, 1]
    )

    # --------------------------------------------------------
    # Apply optimized threshold
    # --------------------------------------------------------

    prediction = (
        prediction_proba >= ml_model["threshold"]
    ).astype(int)

    # --------------------------------------------------------
    # Extract scalar values
    # --------------------------------------------------------

    probability = float(
        prediction_proba[0]
    )

    predicted_class = int(
        prediction[0]
    )

    threshold = float(
        ml_model["threshold"]
    )

    # --------------------------------------------------------
    # Risk interpretation
    # --------------------------------------------------------

    if predicted_class == 1:

        risk = "High Risk"
        decision = "Likely to Default"

    else:

        risk = "Low Risk"
        decision = "Likely to Repay"

    # --------------------------------------------------------
    # Return response
    # --------------------------------------------------------

    return {

        "status": "success",

        "default_probability": round(
            probability,
            4
        ),

        "default_probability_percentage": round(
            probability * 100,
            2
        ),

        "default_prediction": predicted_class,

        "threshold": round(
            threshold,
            4
        ),

        "risk": risk,

        "decision": decision,

        "model": "calibrated_xgb_model.pkl"
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")