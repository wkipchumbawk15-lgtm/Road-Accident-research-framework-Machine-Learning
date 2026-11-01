from fastapi import FastAPI, HTTPException
import joblib
import logging
from pathlib import Path
import pandas as pd

from schemas import PredictRequest, PredictResponse
from config import FEATURE_ORDER
from utils.model_registry import get_active_model, set_active_model, get_models, delete_model_artifact

app = FastAPI(title="Research Accident Classification Prediction Service")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prediction-service")
MODEL_DIR = Path('/models')
model = None
ALGORITHM_NAME = None
NUMERIC_FEATURES = {
    "year", "month", "hour", "minute", "day_of_week", "is_weekend", "age", "no"
}


def load_active_model():
    global model, ALGORITHM_NAME
    try:
        name = get_active_model()
    except Exception:
        model, ALGORITHM_NAME = None, None
        return
    path = MODEL_DIR / f"{name}.joblib"
    if not path.exists():
        model, ALGORITHM_NAME = None, None
        return
    model = joblib.load(path)
    ALGORITHM_NAME = name

@app.on_event("startup")
def startup_event():
    load_active_model()

@app.get("/metadata")
def metadata():
    result = {
        "algorithm": ALGORITHM_NAME,
        "features": FEATURE_ORDER,
        "model_loaded": model is not None,
        "feature_types": {
            feature: "numeric" if feature in NUMERIC_FEATURES else "categorical"
            for feature in FEATURE_ORDER
        },
        "feature_options": {},
        "feature_defaults": {},
        "class_labels": [],
    }
    if model is None:
        return result

    if hasattr(model, "classes_"):
        result["class_labels"] = [str(label) for label in model.classes_]

    preprocess = getattr(model, "named_steps", {}).get("preprocess")
    if preprocess is not None:
        transformers = preprocess.named_transformers_
        numeric_pipeline = transformers.get("num")
        if numeric_pipeline is not None:
            imputer = numeric_pipeline.named_steps.get("imputer")
            if imputer is not None:
                result["feature_defaults"].update(
                    {
                        feature: float(value)
                        for feature, value in zip(
                            numeric_pipeline.feature_names_in_, imputer.statistics_
                        )
                        if feature in FEATURE_ORDER
                    }
                )

        categorical_pipeline = transformers.get("cat")
        if categorical_pipeline is not None:
            imputer = categorical_pipeline.named_steps.get("imputer")
            encoder = categorical_pipeline.named_steps.get("onehot")
            if imputer is not None:
                result["feature_defaults"].update(
                    {
                        feature: str(value)
                        for feature, value in zip(
                            categorical_pipeline.feature_names_in_, imputer.statistics_
                        )
                        if feature in FEATURE_ORDER
                    }
                )
            if encoder is not None:
                result["feature_options"].update(
                    {
                        feature: [str(value) for value in values]
                        for feature, values in zip(
                            categorical_pipeline.feature_names_in_, encoder.categories_
                        )
                        if feature in FEATURE_ORDER
                    }
                )

    return result

@app.get("/models")
def list_models():
    try:
        active = get_active_model()
    except Exception:
        active = None
    return {"models": get_models(), "active_model": active}


@app.delete("/models/{model_name}")
def delete_trained_model(model_name: str):
    global model, ALGORITHM_NAME

    try:
        deleted = delete_model_artifact(model_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    if ALGORITHM_NAME == model_name:
        model, ALGORITHM_NAME = None, None

    return {"status": "success", "deleted_model": model_name}

from pydantic import BaseModel
class ModelSelect(BaseModel):
    model_name: str

@app.post("/models/select")
def select_model(body: ModelSelect):
    try:
        set_active_model(body.model_name)
        load_active_model()
        return {"status": "success", "active_model": body.model_name}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="No active model is loaded.")
    missing = [f for f in FEATURE_ORDER if f not in request.data]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
    invalid_types = [
        feature for feature in FEATURE_ORDER
        if feature not in NUMERIC_FEATURES and not isinstance(request.data[feature], str)
    ]
    if invalid_types:
        raise HTTPException(
            status_code=422,
            detail=f"Categorical features must be text values: {invalid_types}",
        )
    try:
        row = {f: request.data.get(f) for f in FEATURE_ORDER}
        X = pd.DataFrame([row])
        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0].tolist() if hasattr(model, 'predict_proba') else None
        return {
            "prediction": str(pred),
            "probabilities": probs,
            "class_labels": [str(label) for label in model.classes_] if probs is not None else [],
            "algorithm": ALGORITHM_NAME,
            "feature_names": FEATURE_ORDER,
        }
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))
