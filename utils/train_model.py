"""Research-aligned supervised learning pipeline.

The classifier is deliberately target-agnostic because the NTSA dataset does not
contain an accident_severity field. The target must therefore be supplied by the
research workflow after methodological approval.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from utils.model_registry import MODEL_DIR, register_model, set_active_model
from utils.research_schema import candidate_predictors, engineer_ntsa_features, normalize_column_name

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
STATUS_FILE = DATA_DIR / "evaluation_results.json"


def _ensure_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reconcile raw NTSA data with the supervised training schema.

    The preprocessing stage writes a cleaned dataset that should already contain
    engineered fields such as `hour`, `minute`, and `day_of_week`. Some datasets,
    however, still arrive in a semi-raw format or omit one of these derived
    columns. In those cases, apply the canonical research feature engineering
    pass and backfill any still-missing required features with safe defaults.
    """
    required_numeric = {
        "year", "month", "hour", "minute", "day_of_week", "is_weekend", "age", "no"
    }
    required_categorical = {
        "county", "road", "place", "vehicle_type", "gender", "age_group", "victim", "time_period"
    }

    if any(c in df.columns for c in ["date", "time_24_hours", "age", "county", "vehicle_type", "gender"]) or not required_numeric.issubset(df.columns):
        df = engineer_ntsa_features(df)

    for col, default in {
        "year": 2000,
        "month": 1,
        "hour": 0,
        "minute": 0,
        "day_of_week": 0,
        "is_weekend": 0,
        "age": 0,
        "no": 0,
    }.items():
        if col not in df.columns:
            df[col] = default

    for col in ["county", "road", "place", "vehicle_type", "gender", "age_group", "victim", "time_period"]:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("MISSING").str.strip().str.upper()
        else:
            df[col] = "MISSING"

    if "age_group" in df.columns:
        df["age_group"] = df["age_group"].astype("string").fillna("UNKNOWN")
    if "time_period" in df.columns:
        df["time_period"] = df["time_period"].astype("string").fillna("UNKNOWN")

    return df


def _write_status(payload):
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(payload, default=str, indent=2))


def load_dataset(dataset_path: str, target_column: str):
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Research dataset not found: {path}")
    df = pd.read_csv(path)
    df.columns = [normalize_column_name(c) for c in df.columns]
    df = _ensure_engineered_features(df)
    target = normalize_column_name(target_column)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found. Available columns: {list(df.columns)}")

    if target in {"date", "time_24_hours", "day", "year", "month", "hour", "minute", "no"}:
        raise ValueError(
            "DATE is a date/time or numeric field, not a class label. "
            "The proposal-aligned classification target is 'cause_category'; DATE is used only "
            "to derive temporal predictors."
        )

    df = df.dropna(subset=[target]).copy()
    unique_count = int(df[target].nunique(dropna=True))
    if unique_count < 2:
        raise ValueError("Classification requires at least two target classes.")
    if unique_count > 25:
        raise ValueError(
            f"Target '{target}' has {unique_count} unique values and is not suitable for classification. "
            "Choose a research-approved target with no more than 25 classes. Raw cause codes may need to be grouped "
            "into a documented category target before training."
        )

    predictors = candidate_predictors(df, target)
    if not predictors:
        raise ValueError("No supported predictor fields remain after excluding the target.")

    X = df[predictors].copy()
    y = df[target].astype(str)
    return X, y, predictors, len(df)


def _pipeline(model):
    numeric = ["year", "month", "hour", "minute", "day_of_week", "is_weekend", "age", "no"]
    categorical = ["county", "road", "place", "vehicle_type", "gender", "age_group", "victim", "time_period"]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ], remainder="drop")
    return Pipeline([
        ("preprocess", preprocessor),
        ("select", SelectPercentile(score_func=f_classif, percentile=80)),
        ("classifier", model),
    ])


def train_all_models(dataset_path: str, target_column: str, classifiers: Iterable[str] | None = None):
    classifiers = [c.lower() for c in (classifiers or ["random_forest", "svm"])]
    allowed = {"random_forest", "svm"}
    unknown = [c for c in classifiers if c not in allowed]
    if unknown:
        raise ValueError(f"Unsupported research classifiers: {unknown}. Supported: {sorted(allowed)}")

    X, y, predictors, rows_used = load_dataset(dataset_path, target_column)
    # Stratification is retained, but classes with only one observation cannot be split.
    counts = y.value_counts()
    rare = counts[counts < 2].index
    if len(rare):
        keep = ~y.isin(rare)
        X, y = X.loc[keep], y.loc[keep]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    n_splits = min(5, int(y_train.value_counts().min()))
    if n_splits < 2:
        raise ValueError("Not enough observations per class for stratified cross-validation.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    models = {
        "random_forest": (
            _pipeline(RandomForestClassifier(random_state=42, n_jobs=1, class_weight="balanced")),
            {"classifier__n_estimators": [100], "classifier__max_depth": [None, 15]}
        ),
        "svm": (
            _pipeline(LinearSVC(
                random_state=42,
                class_weight="balanced",
                dual="auto",
                max_iter=20000,
            )),
            {"classifier__C": [0.5, 1.0]}
        ),
    }

    results = {}
    best_name, best_cv = None, -1.0
    for name in classifiers:
        pipeline, grid = models[name]
        _write_status({"status": "running", "step": name, "target": target_column})
        gs = GridSearchCV(pipeline, grid, scoring="f1_weighted", cv=cv, n_jobs=1, refit=True)
        gs.fit(X_train, y_train)
        pred = gs.best_estimator_.predict(X_test)
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, average="weighted", zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, average="weighted", zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, pred, average="weighted", zero_division=0)), 4),
        }
        register_model(name, gs.best_estimator_)
        results[name] = {
            "cv_f1_weighted": round(float(gs.best_score_), 4),
            "best_params": gs.best_params_,
            "test_metrics": metrics,
        }
        if gs.best_score_ > best_cv:
            best_name, best_cv = name, float(gs.best_score_)

    if best_name:
        set_active_model(best_name)

    feature_order = {"target": normalize_column_name(target_column), "predictors": predictors}
    (MODEL_DIR / "feature_order.json").write_text(json.dumps(feature_order, indent=2))
    return {
        "status": "success",
        "target_column": normalize_column_name(target_column),
        "rows_used": int(rows_used),
        "predictors": predictors,
        "models": list(results),
        "results": results,
        "best_model": best_name,
        "best_cv_score": round(best_cv, 4),
    }
