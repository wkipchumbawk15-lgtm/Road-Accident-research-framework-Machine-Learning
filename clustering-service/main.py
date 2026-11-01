from fastapi import FastAPI, Request, HTTPException
import logging
import pandas as pd
from utils.clustering_engine import run_clustering, SUPPORTED_ALGORITHMS, DEFAULT_FEATURES
from utils.feature_engineering import CATEGORICAL_FEATURES, HIGH_CARDINALITY_FEATURES, NUMERIC_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("clustering-service")

app = FastAPI(title="NTSA Research Clustering Service")

@app.post("/run-cluster")
async def run_cluster(request: Request):
    try:
        payload = await request.json()
        df = pd.DataFrame(payload.get("data", payload) if isinstance(payload, dict) else payload)

        # Normalize common column name variants to canonical feature names
        def _canonical_col(col: str) -> str:
            c = str(col).lower().strip()
            c = c.replace('.', '').replace('-', ' ').replace('/', ' ').replace('%', ' ')
            c = ' '.join(c.split())
            # heuristics
            if 'county' in c:
                return 'county'
            if 'vehicle' in c and 'type' in c:
                return 'vehicle_type'
            if c in {'type of vehicle', 'type of vehicle'}:
                return 'vehicle_type'
            if 'gender' in c:
                return 'gender'
            if 'victim' in c:
                return 'victim'
            if 'age' == c or c.startswith('age ' ) or c == 'age':
                return 'age'
            if 'time' in c and '24' in c:
                return 'time_24_hours'
            if c in {'no', 'no', 'no.'} or c.endswith(' no') or c == 'no':
                return 'no'
            if 'road' in c:
                return 'road'
            if 'place' in c:
                return 'place'
            if 'date' in c:
                return 'date'
            if c in {'day'}:
                return 'day'
            # fallback: return original lower without spaces
            return c.replace(' ', '_')

        rename_map = {}
        for col in df.columns:
            canon = _canonical_col(col)
            # Only rename when canonical differs to avoid clobbering engineered names
            if canon != str(col):
                rename_map[col] = canon
        if rename_map:
            logger.info("Renaming incoming columns: %s", rename_map)
            df = df.rename(columns=rename_map)

        # Coerce known numeric features to numeric, invalid parsing -> NaN
        try:
            numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
            if numeric_cols:
                logger.info("Coercing numeric columns to numeric types: %s", numeric_cols)
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception:
            logger.debug("Numeric coercion failed or not applicable")

        # Diagnostic logging: report NA counts and a small sample to help debug
        try:
            na_counts = df.isna().sum().to_dict()
        except Exception:
            na_counts = {}

        logger.info("Received clustering payload: rows=%d, columns=%s", len(df), list(df.columns))

        # Defensive imputation: only fill known categorical/high-cardinality features
        try:
            categorical_cols = [c for c in (CATEGORICAL_FEATURES + HIGH_CARDINALITY_FEATURES) if c in df.columns]
            for col in categorical_cols:
                df[col] = df[col].astype('string').fillna('MISSING')
            logger.info("Imputed categorical/high-cardinality columns: %s", categorical_cols)
        except Exception:
            logger.debug("Failed to impute categorical columns before clustering")

        logger.info("Column NA counts: %s", na_counts)
        try:
            logger.debug("Data sample for clustering: %s", df.head(5).to_dict(orient="records"))
        except Exception:
            logger.debug("Data sample unavailable for logging")

        algorithms = payload.get("algorithms") if isinstance(payload, dict) else None
        features = payload.get("features") if isinstance(payload, dict) else None

        try:
            result = run_clustering(df, algorithms=algorithms, features=features)
        except Exception as exc:
            # Log full traceback for diagnostics, then re-raise as HTTP 502
            logger.exception("Clustering call failed")
            raise HTTPException(status_code=502, detail=str(exc))

        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.get("/algorithms")
def algorithms():
    return {"algorithms": SUPPORTED_ALGORITHMS, "default_features": DEFAULT_FEATURES}
