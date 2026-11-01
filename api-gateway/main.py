from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
import httpx
import os
import json
import logging
import io
import pandas as pd
from datetime import datetime
import numpy as np
from typing import Any

from utils.feature_engineering import NUMERIC_FEATURES, CATEGORICAL_FEATURES, HIGH_CARDINALITY_FEATURES

from utils.model_registry import set_active_model
from utils.train_model import train_all_models
from utils.model_registry import get_active_model

from utils.model_registry import set_active_model
from utils.model_registry import get_active_model

# =========================================================
# Service URLs (resolved via Docker network)
# =========================================================
PREDICTION_URL = os.getenv("PREDICTION_URL", "http://prediction:8000")
CLUSTER_URL = os.getenv("CLUSTER_URL", "http://clustering:8000")

TRAINING_URL = os.getenv("TRAINING_URL", "http://ml-training:8000")
PREPROCESS_URL = os.getenv("PREPROCESS_URL", "http://preprocessing:8000")



EVAL_FILE = "/data/evaluation_results.json"

# Expected features used by clustering (for quick client-side validation)
EXPECTED_CLUSTER_FEATURES = [
    "age", "hour", "month", "day_of_week", "is_weekend", "no",
    "county", "vehicle_type", "gender", "victim", "age_group", "time_period"
]

# =========================================================
# App & Logging
# =========================================================
app = FastAPI(
    title="Road Accident Modelling API Gateway",
    version="1.0.0",
    description="Central entry point for preprocessing, training, clustering, and prediction services"
)

from utils.cluster_job import cluster_job

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("api-gateway")

# =========================================================
# MinIO Configuration (Lazy Initialization)
# =========================================================
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET = "uploads"

_minio_client = None

def get_minio_client():
    """Lazy initialize MinIO client only when needed"""
    global _minio_client
    if _minio_client is None:
        try:
            from minio import Minio
            _minio_client = Minio(
                MINIO_ENDPOINT,
                access_key=MINIO_ACCESS,
                secret_key=MINIO_SECRET,
                secure=False
            )
            # Create bucket if it doesn't exist
            if not _minio_client.bucket_exists(BUCKET):
                _minio_client.make_bucket(BUCKET)
                logger.info(f"Created bucket: {BUCKET}")
        except Exception as e:
            logger.error(f"Failed to initialize MinIO: {e}")
            _minio_client = None
    return _minio_client

# =========================================================
# Utility
# =========================================================
def safe_json_response(response: httpx.Response):
    logger.info(
        "Downstream response status=%s content-type=%s",
        response.status_code,
        response.headers.get("content-type")
    )
    logger.debug("Downstream response body: %s", response.text)

    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail=response.text or "Downstream service error"
        )

    if "application/json" not in response.headers.get("content-type", ""):
        raise HTTPException(
            status_code=502,
            detail="Invalid response type from downstream service"
        )

    return response.json()


def find_cleaned_object_name(minio_client, object_name: str):
    """Resolve the latest cleaned object matching a raw object name."""
    if minio_client is None or not object_name:
        return None

    exact_key = f"cleaned_{object_name}"
    candidate_keys = {
        exact_key,
        exact_key.replace(" ", "%20"),
        exact_key.replace(" ", "+")
    }

    try:
        for obj in minio_client.list_objects(BUCKET, recursive=True):
            key = obj.object_name
            normalized_key = key.replace("%20", " ").replace("+", " ")
            if key in candidate_keys or normalized_key in candidate_keys:
                logger.info("Matched cleaned object by exact key: %s", key)
                return key
    except Exception as exc:
        logger.debug("Exact cleaned object search failed for %s: %s", object_name, exc)

    try:
        base, _ = os.path.splitext(object_name)
        prefix = f"cleaned_{base}"
        candidates = []
        for obj in minio_client.list_objects(BUCKET, recursive=True):
            if obj.object_name.startswith(prefix):
                candidates.append(obj)

        if candidates:
            from datetime import datetime
            latest = max(candidates, key=lambda o: getattr(o, "last_modified", datetime.min))
            logger.info("Matched cleaned object by prefix: %s", latest.object_name)
            return latest.object_name
    except Exception as exc:
        logger.debug("Prefix cleaned object search failed for %s: %s", object_name, exc)

    try:
        metadata_key = f"metadata_{object_name}.json"
        obj = minio_client.get_object(BUCKET, metadata_key)
        metadata = json.loads(obj.read().decode("utf-8"))
        cleaned_name = metadata.get("cleaned_object")
        if cleaned_name:
            logger.info("Resolved cleaned object from metadata: %s", cleaned_name)
            return cleaned_name
    except Exception as exc:
        logger.debug("Metadata lookup failed for %s: %s", object_name, exc)

    return None

# =========================================================
# Request Models
# =========================================================
class ClusterRequest(BaseModel):
    object_name: str

class PredictRequest(BaseModel):
    data: dict = Field(
        ...,
        description="Feature names mapped to numeric values",
        example={
            "age": 45,
            "hour": 18,
            "month": 6,
            "day_of_week": 2,
            "county": "NAIROBI",
            "road": "MOMBASA ROAD",
            "place": "CITY CENTRE",
            "vehicle_type": "PRIVATE",
            "gender": "M",
            "victim": "DRV",
            "no": 1
        }
    )

class TrainRequest(BaseModel):
    dataset_object: str
    target_column: str
    classifiers: list = Field(default=["random_forest", "svm"])
    clusterers: list = Field(default=["kmeans", "dbscan", "agglomerative", "mean_shift"])

# =========================================================
# Health
# =========================================================
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "API Gateway running"}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}

# =========================================================
# Preprocessing
# =========================================================
@app.post("/upload", tags=["Preprocessing"])
async def upload_dataset(file: UploadFile = File(...)):
    async with httpx.AsyncClient(timeout=120) as client:
        files = {"file": (file.filename, await file.read())}
        response = await client.post(f"{PREPROCESS_URL}/upload", files=files)
        print("Preprocessing upload response:", response.status_code, response.text)
        return safe_json_response(response)

@app.get("/preprocess/status/{job_id}", tags=["Preprocessing"])
def preprocess_status(job_id: str):
    response = httpx.get(f"{PREPROCESS_URL}/status/{job_id}", timeout=10)
    return safe_json_response(response)

# @app.get("/list-files", tags=["Preprocessing"])
# def list_uploaded_files():
#     """List all uploaded files in MinIO bucket"""
#     try:
#         minio_client = get_minio_client()
#         if minio_client is None:
#             raise Exception("MinIO client not initialized")
        
#         files = []
#         objects = minio_client.list_objects(BUCKET)
#         for obj in objects:
#             # Skip cleaned files from the list (show only original uploads)
#             if not obj.object_name.startswith("cleaned_"):
#                 files.append(obj.object_name)
#         logger.info(f"Listed {len(files)} files from bucket")
#         return {"status": "success", "files": sorted(files)}
#     except Exception as e:
#         logger.error(f"Error listing files: {e}")
#         raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")

@app.get("/list-files", tags=["Preprocessing"])
def list_uploaded_files():
    """List all uploaded files in MinIO bucket"""
    try:
        minio_client = get_minio_client()

        if minio_client is None:
            raise Exception("MinIO client not initialized")

        files = []

        # IMPORTANT FIX
        objects = minio_client.list_objects(
            BUCKET,
            recursive=True
        )

        for obj in objects:
            logger.info(f"Found object: {obj.object_name}")

            # Skip cleaned files from dropdown
            if not obj.object_name.startswith("cleaned_"):
                files.append(obj.object_name)

        logger.info(f"Listed {len(files)} files from bucket")

        return {
            "status": "success",
            "files": sorted(files)
        }

    except Exception as e:
        logger.error(f"Error listing files: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"Error listing files: {str(e)}"
        )


@app.get("/inspect-columns/{object_name}", tags=["Preprocessing"])
def inspect_columns(object_name: str):
    """Return columns found in the uploaded CSV and compare to expected clustering features."""
    try:
        minio_client = get_minio_client()
        if minio_client is None:
            raise HTTPException(status_code=503, detail="MinIO client not initialized")

        raw_obj = minio_client.get_object(BUCKET, object_name)
        raw_bytes = raw_obj.read()
        _, ext = os.path.splitext(object_name)
        ext = ext.lower()

        # Choose reader based on file extension
        if ext in {".xlsx", ".xls"}:
            try:
                raw_df = pd.read_excel(io.BytesIO(raw_bytes), nrows=0)
            except Exception:
                # Try without nrows for older pandas versions, then take columns
                raw_df = pd.read_excel(io.BytesIO(raw_bytes))
                raw_df = raw_df.head(0)
        else:
            raw_df = pd.read_csv(io.BytesIO(raw_bytes), nrows=0)

        raw_columns = [c for c in raw_df.columns]

        cleaned_key = find_cleaned_object_name(minio_client, object_name)
        cleaned_columns = None
        if cleaned_key:
            cleaned_obj = minio_client.get_object(BUCKET, cleaned_key)
            cleaned_bytes = cleaned_obj.read()
            _, cext = os.path.splitext(cleaned_key)
            cext = cext.lower()
            if cext in {".xlsx", ".xls"}:
                try:
                    cleaned_df = pd.read_excel(io.BytesIO(cleaned_bytes), nrows=0)
                except Exception:
                    cleaned_df = pd.read_excel(io.BytesIO(cleaned_bytes))
                    cleaned_df = cleaned_df.head(0)
            else:
                cleaned_df = pd.read_csv(io.BytesIO(cleaned_bytes), nrows=0)
            cleaned_columns = [c for c in cleaned_df.columns]

        expected = EXPECTED_CLUSTER_FEATURES
        missing = [c for c in expected if c not in (cleaned_columns or raw_columns)]
        extra = [c for c in (cleaned_columns or raw_columns) if c not in expected]

        return {
            "status": "success",
            "object_name": object_name,
            "cleaned_object": cleaned_key,
            "uploaded_columns": raw_columns,
            "cleaned_columns": cleaned_columns,
            "expected_columns": expected,
            "missing_expected_columns": missing,
            "extra_columns": extra
        }

    except Exception as e:
        logger.error(f"Error inspecting file columns for {object_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clean-and-verify/{object_name}", tags=["Preprocessing"])
def clean_and_verify(object_name: str):
    """Run lightweight cleaning locally and persist a cleaned CSV to MinIO.

    Returns cleaned object key, column list, NA counts and a small sample.
    """
    try:
        minio_client = get_minio_client()
        if minio_client is None:
            raise HTTPException(status_code=503, detail="MinIO client not initialized")

        # Prefer cleaned version if present
        cleaned_key = find_cleaned_object_name(minio_client, object_name)
        if cleaned_key:
            obj = minio_client.get_object(BUCKET, cleaned_key)
            raw_bytes = obj.read()
            logger.info("Using existing cleaned object: %s", cleaned_key)
            base_name = cleaned_key
        else:
            obj = minio_client.get_object(BUCKET, object_name)
            raw_bytes = obj.read()
            base_name = object_name

        _, ext = os.path.splitext(base_name)
        ext = ext.lower()
        if ext in {".xlsx", ".xls"}:
            try:
                df = pd.read_excel(io.BytesIO(raw_bytes))
            except Exception:
                df = pd.read_excel(io.BytesIO(raw_bytes))
        else:
            df = pd.read_csv(io.BytesIO(raw_bytes))

        # Normalize incoming column names to expected canonical names before cleaning
        def _canonical_col(col: str) -> str:
            c = str(col).lower().strip()
            c = c.replace('.', '').replace('-', ' ').replace('/', ' ').replace('%', ' ')
            c = ' '.join(c.split())
            if 'county' in c:
                return 'county'
            if 'vehicle' in c and 'type' in c:
                return 'vehicle_type'
            if 'type of vehicle' in c or 'type of vehicle' == c:
                return 'vehicle_type'
            if 'gender' in c:
                return 'gender'
            if 'victim' in c:
                return 'victim'
            if c == 'age' or c.startswith('age '):
                return 'age'
            if 'time' in c and '24' in c:
                return 'time_24_hours'
            if c in {'no', 'no.'} or c.endswith(' no'):
                return 'no'
            if 'road' in c:
                return 'road'
            if 'place' in c:
                return 'place'
            if 'date' in c:
                return 'date'
            if c == 'day':
                return 'day'
            return col

        rename_map = {}
        for col in df.columns:
            canon = _canonical_col(col)
            if canon != col:
                rename_map[col] = canon
        if rename_map:
            logger.info("Renaming columns for cleaning: %s", rename_map)
            df = df.rename(columns=rename_map)

        # Coerce known numeric features on the raw df to numeric (invalid -> NaN)
        try:
            num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
            for col in num_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            logger.info("Coerced numeric columns before cleaning: %s", num_cols)
        except Exception:
            logger.debug("Numeric coercion before cleaning failed")

        # Run engineered feature cleaning
        try:
            from utils.feature_engineering import make_research_features
            cleaned_df = make_research_features(df)
        except Exception as e:
            logger.exception("Feature engineering failed")
            raise HTTPException(status_code=500, detail=f"Feature engineering failed: {e}")

        # Ensure expected categorical features are filled and present
        try:
            cat_cols = [c for c in (CATEGORICAL_FEATURES + HIGH_CARDINALITY_FEATURES) if c in cleaned_df.columns]
            for col in cat_cols:
                cleaned_df[col] = cleaned_df[col].astype('string').fillna('MISSING')
            if 'county' not in cleaned_df.columns:
                cleaned_df['county'] = 'MISSING'
            logger.info("Post-cleaning ensured categorical columns: %s", cat_cols)
        except Exception:
            logger.debug("Post-cleaning categorical fill failed")

        # Persist cleaned CSV with timestamped key
        base, _ = os.path.splitext(object_name)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        cleaned_name = f"cleaned_{base}_{ts}.csv"
        out = cleaned_df.to_csv(index=False).encode("utf-8")
        try:
            minio_client.put_object(BUCKET, cleaned_name, io.BytesIO(out), length=len(out), content_type="text/csv")
            logger.info("Wrote cleaned object: %s (%d bytes)", cleaned_name, len(out))
        except Exception as e:
            logger.exception("Failed to write cleaned object to MinIO")
            raise HTTPException(status_code=500, detail=f"Failed to write cleaned object: {e}")

        # Prepare verification info
        na_counts = cleaned_df.isna().sum().to_dict()
        sample = cleaned_df.head(5).to_dict(orient="records")
        cols = list(cleaned_df.columns)

        # Sanitize for JSON: convert numpy/pandas types and remove infinities
        def _sanitize_value(v):
            try:
                if pd.isna(v):
                    return None
            except Exception:
                pass
            try:
                if isinstance(v, (np.floating, float)):
                    if np.isfinite(v):
                        return float(v)
                    return None
                if isinstance(v, (np.integer, int)):
                    return int(v)
            except Exception:
                pass
            try:
                if hasattr(v, 'item'):
                    return v.item()
            except Exception:
                pass
            return v

        sanitized_sample = []
        for rec in sample:
            cleaned_rec = {k: _sanitize_value(v) for k, v in rec.items()}
            sanitized_sample.append(cleaned_rec)

        sanitized_na = {k: int(v) if pd.notna(v) else None for k, v in na_counts.items()}

        return {
            "status": "success",
            "cleaned_object": cleaned_name,
            "columns": cols,
            "na_counts": sanitized_na,
            "sample": sanitized_sample
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("clean_and_verify failed for %s", object_name)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/register-clustering/{object_name}", tags=["Clustering"])
def register_clustering(object_name: str):
    """Run clustering on the cleaned object, persist result to MinIO and update models/versions.json.

    Returns the MinIO key for the clustering JSON and the new version id.
    """
    try:
        minio_client = get_minio_client()
        if minio_client is None:
            raise HTTPException(status_code=503, detail="MinIO client not initialized")

        # Resolve cleaned object
        cleaned_key = find_cleaned_object_name(minio_client, object_name) or object_name
        try:
            obj = minio_client.get_object(BUCKET, cleaned_key)
            df_bytes = obj.read()
        except Exception as e:
            logger.exception("Failed to read cleaned object for clustering: %s", cleaned_key)
            raise HTTPException(status_code=404, detail=f"Cleaned object not found: {cleaned_key}")

        # Load dataframe and prepare records
        try:
            df = pd.read_csv(io.BytesIO(df_bytes))
        except Exception:
            try:
                df = pd.read_excel(io.BytesIO(df_bytes))
            except Exception as e:
                logger.exception("Failed to parse cleaned object: %s", cleaned_key)
                raise HTTPException(status_code=400, detail=f"Failed to parse cleaned object: {e}")

        data = df.to_dict(orient='records')

        # Clean NaN/inf for JSON
        import math
        for rec in data:
            for k, v in list(rec.items()):
                if isinstance(v, float):
                    if math.isnan(v) or math.isinf(v):
                        rec[k] = None

        # POST to clustering service
        try:
            response = httpx.post(f"{CLUSTER_URL}/run-cluster", json=data, timeout=600)
        except Exception as e:
            logger.exception("Failed to call clustering service")
            raise HTTPException(status_code=502, detail=str(e))

        if response.status_code >= 400:
            logger.error("Clustering service returned error: %s", response.text)
            raise HTTPException(status_code=response.status_code, detail=response.text)

        clustering_result = response.json()

        # Persist clustering result to MinIO
        base, _ = os.path.splitext(cleaned_key)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        clustering_key = f"clustering_{base}_{ts}.json"
        payload = json.dumps(clustering_result).encode('utf-8')
        try:
            minio_client.put_object(BUCKET, clustering_key, io.BytesIO(payload), length=len(payload), content_type='application/json')
            logger.info("Wrote clustering result to MinIO: %s", clustering_key)
        except Exception as e:
            logger.exception("Failed to write clustering result to MinIO")
            raise HTTPException(status_code=500, detail=f"Failed to write clustering result: {e}")

        # Update models/versions.json by creating a new version entry
        try:
            versions_path = os.path.join('models', 'versions.json')
            versions = {}
            if os.path.exists(versions_path):
                with open(versions_path, 'r', encoding='utf-8') as f:
                    versions = json.load(f)

            # Determine next version id
            existing = [int(k.lstrip('v')) for k in versions.keys() if k.lstrip('v').isdigit()]
            next_v = max(existing) + 1 if existing else 1
            version_name = f"v{next_v}"

            # Build minimal version metadata
            clusterers = [r.get('algorithm') for r in clustering_result.get('results', [])]
            version_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'classifiers': [],
                'clusterers': clusterers,
                'best_classifier': None,
                'classification_results': {},
                'clustering_results': clustering_result
            }

            versions[version_name] = version_entry
            with open(versions_path, 'w', encoding='utf-8') as f:
                json.dump(versions, f, indent=2)
            logger.info("Updated versions manifest with %s", version_name)
        except Exception as e:
            logger.exception("Failed to update versions manifest")
            raise HTTPException(status_code=500, detail=f"Failed to update versions manifest: {e}")

        return {
            'status': 'success',
            'version': version_name,
            'clustering_key': clustering_key
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("register_clustering failed for %s", object_name)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{object_name}", tags=["Preprocessing"])
def download_file(object_name: str):
    """Download a cleaned or raw CSV file from MinIO.

    Handles common filename encodings (spaces, %20, +) by attempting
    several candidate object names and logging detailed failures.
    """
    from fastapi.responses import StreamingResponse

    minio_client = get_minio_client()
    if minio_client is None:
        raise HTTPException(status_code=503, detail="MinIO service not available")

    cleaned_key = find_cleaned_object_name(minio_client, object_name)
    candidates = []
    if cleaned_key:
        candidates.append(cleaned_key)
    candidates.extend([
        object_name,
        object_name.replace("%20", " "),
        object_name.replace("+", " ")
    ])

    last_exc = None
    for name in candidates:
        try:
            logger.info(f"Trying download candidate: {name}")
            # Try to get the object directly (stat may behave differently across MinIO versions)
            obj = minio_client.get_object(BUCKET, name)
            data = obj.read()
            logger.info(f"Successfully read object: {name} ({len(data)} bytes)")
            return StreamingResponse(
                iter([data]),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={name}"}
            )
        except Exception as e:
            logger.debug(f"Download candidate failed: {name} -> {e}")
            last_exc = e

    # If we reach here no candidate succeeded
    logger.error(f"All download candidates failed for {object_name}: last error: {last_exc}")
    raise HTTPException(status_code=404, detail=f"File not found: {object_name} (tried {len(candidates)} candidates)")


@app.get("/download-by-key/{object_name}", tags=["Preprocessing"])
def download_by_key(object_name: str):
    """Find the best-matching object key using `list_objects` and stream it.

    This searches bucket keys for a candidate that matches after normalizing
    common encodings (spaces, %20, +) and returns the object bytes.
    """
    from fastapi.responses import StreamingResponse

    minio_client = get_minio_client()
    if minio_client is None:
        raise HTTPException(status_code=503, detail="MinIO service not available")

    # Build candidate forms to match against object keys
    candidates = set()
    cleaned = f"cleaned_{object_name}"
    candidates.update([object_name, cleaned])
    candidates.update([c.replace("%20", " ").replace("+", " ") for c in list(candidates)])

    logger.info(f"Searching for object matching candidates: {candidates}")

    try:
        for obj in minio_client.list_objects(BUCKET, recursive=True):
            key = obj.object_name
            norm_key = key.replace("%20", " ").replace("+", " ")
            logger.info(f"Checking object key: {key} (normalized: {norm_key})")
            for cand in candidates:
                logger.debug(f"Comparing candidate '{cand}' with normalized key '{norm_key}'")
                if norm_key == cand:
                    logger.info(f"Matched object key: {key}")
                    o = minio_client.get_object(BUCKET, key)
                    data = o.read()
                    return StreamingResponse(
                        iter([data]),
                        media_type="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={key}"}
                    )
        logger.error(f"No matching object found for {object_name}")
        raise HTTPException(status_code=404, detail=f"No matching object found for {object_name}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error during download-by-key for {object_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/presigned-download/{object_name}", tags=["Preprocessing"])
def presigned_download(object_name: str):
    """Return a presigned GET URL for the cleaned or raw object.

    The presigned URL is generated using `localhost:9000` as host so it
    is directly accessible from the developer machine where Docker is
    running. This avoids needing DNS resolution for the internal host
    name `minio`.
    """
    try:
        from datetime import timedelta
        # Build candidate names to try
        minio_client = get_minio_client()
        if minio_client is None:
            raise HTTPException(status_code=503, detail="MinIO client not initialized")

        found_key = find_cleaned_object_name(minio_client, object_name) or object_name
        try:
            minio_client.stat_object(BUCKET, found_key)
        except Exception:
            raise HTTPException(status_code=404, detail=f"No matching object for {object_name}")

        if not found_key:
            raise HTTPException(status_code=404, detail=f"No matching object for {object_name}")

        # Create a Minio client configured with localhost so the generated
        # presigned URL uses a host reachable from the developer machine.
        from minio import Minio as LocalMinio
        local_client = LocalMinio('localhost:9000', access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=False)

        url = local_client.presigned_get_object(BUCKET, found_key, expires=timedelta(seconds=3600))
        return {"status": "success", "object_key": found_key, "url": url}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to create presigned URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/files/{object_name}", tags=["Preprocessing"])
def delete_uploaded_file(object_name: str):
    try:
        minio_client = get_minio_client()
        if minio_client is None:
            raise HTTPException(status_code=503, detail="MinIO service not available")

        deleted = []
        candidates = [object_name, f"cleaned_{object_name}"]
        for name in candidates:
            try:
                minio_client.stat_object(BUCKET, name)
                minio_client.remove_object(BUCKET, name)
                deleted.append(name)
                logger.info("Deleted object: %s", name)
            except Exception:
                logger.info("Object not found or not deleted: %s", name)

        if not deleted:
            raise HTTPException(status_code=404, detail=f"No file found to delete for {object_name}")

        return {
            "status": "success",
            "deleted": deleted
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file {object_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

# =========================================================
# Clustering
# =========================================================
@app.post("/run-cluster", tags=["Clustering"])
def run_clustering(payload: ClusterRequest):
    object_name = payload.object_name
    minio_client = get_minio_client()
    if minio_client is None:
        raise HTTPException(status_code=503, detail="MinIO service not available")

    # If caller already provided a cleaned object key, use it directly when present.
    cleaned_key = None
    try:
        if object_name and object_name.startswith("cleaned_"):
            try:
                minio_client.stat_object(BUCKET, object_name)
                cleaned_key = object_name
                logger.info("Using provided cleaned object key: %s", object_name)
            except Exception:
                # Provided cleaned key not found; fall through to resolution below
                logger.debug("Provided cleaned key not found in bucket: %s", object_name)

        if cleaned_key is None:
            # Prefer exact cleaned file name (handles simple cases), otherwise try
            # to find the most recent cleaned version matching the base filename.
            cleaned_key = find_cleaned_object_name(minio_client, object_name)
    except Exception as exc:
        logger.debug("Error while resolving cleaned object key: %s", exc)
    df = None

    if cleaned_key:
        try:
            obj = minio_client.get_object(BUCKET, cleaned_key)
            df = pd.read_csv(io.BytesIO(obj.read()))
            logger.info(f"Using cleaned object: {cleaned_key}")
        except Exception as exc:
            logger.error("Failed to read cleaned object %s: %s", cleaned_key, exc)

    if df is None:
        try:
            obj = minio_client.get_object(BUCKET, object_name)
            df = pd.read_csv(io.BytesIO(obj.read()))
            logger.warning("Cleaned object not found; falling back to raw file %s", object_name)
        except Exception as e2:
            logger.error("Neither cleaned nor raw file found: %s", e2)
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {object_name} or cleaned versions. Please upload data first via the 'Upload Dataset' button."
            )
    
    try:
        data = df.to_dict('records')
    except Exception as e:
        logger.error(f"Error converting dataframe to records: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
    try:
        # Replace NaN and inf with null for JSON serialization
        import math
        for record in data:
            for key, value in record.items():
                if isinstance(value, float):
                    if math.isnan(value) or math.isinf(value):
                        record[key] = None
        
        logger.info(
            "Expected clustering request payload: list of record objects from %s; sending %d records",
            object_name,
            len(data)
        )
        logger.debug("Clustering request payload sample: %s", json.dumps(data[:3], default=str))

        # Increase timeout for long-running clustering operations
        response = httpx.post(
            f"{CLUSTER_URL}/run-cluster",
            json=data,
            timeout=600
        )

        logger.info(
            "Received clustering service response status=%s content-type=%s",
            response.status_code,
            response.headers.get("content-type")
        )
        logger.debug("Clustering service response body (truncated): %s", response.text[:1000])

        # If error status, surface it via safe_json_response for consistent handling
        if response.status_code >= 400:
            return safe_json_response(response)

        # If the payload is large, persist to MinIO and return a small reference
        try:
            content_bytes = response.content
            size = len(content_bytes)
            SIZE_THRESHOLD = int(os.getenv('CLUSTER_RESPONSE_SIZE_THRESHOLD', 1_000_000))
            if size > SIZE_THRESHOLD and minio_client is not None:
                base, _ = os.path.splitext(object_name)
                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                clustering_key = f"clustering_{base}_{ts}.json"
                try:
                    minio_client.put_object(BUCKET, clustering_key, io.BytesIO(content_bytes), length=size, content_type='application/json')
                    logger.info("Stored large clustering result to MinIO: %s (%d bytes)", clustering_key, size)
                    return {"status": "stored", "clustering_key": clustering_key}
                except Exception as e:
                    logger.exception("Failed to persist large clustering result to MinIO: %s", e)
                    # Fall through to attempting to return JSON

            # Try to parse and return JSON inline (small payload or persistence failed)
            try:
                return response.json()
            except Exception:
                # If parsing fails, return raw text safely
                return {"status": "ok", "raw": response.text}
        except Exception as e:
            logger.exception("Error handling clustering response: %s", e)
            raise HTTPException(status_code=500, detail=str(e))
        
    except Exception as e:
        logger.error(f"Clustering service error: {e}")
        raise HTTPException(status_code=502, detail=f"Clustering service error: {str(e)}")

# =========================================================
# Training
# =========================================================


@app.post("/train-models", tags=["Training"])
def train_models(request: TrainRequest = None):
    try:
        # Use provided request or create default
        if request is None:
            request = TrainRequest()
        
        # Enqueue background training job via RQ and return the job id.
        try:
            from redis import Redis
            from rq import Queue
            from utils.train_job import train_job

            redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
            redis_conn = Redis.from_url(redis_url)
            q = Queue("training", connection=redis_conn)
            job = q.enqueue(train_job, dataset_object=request.dataset_object, target_column=request.target_column, classifiers=request.classifiers, clusterers=request.clusterers, job_timeout=3600)

            return {
                "status": "queued",
                "job_id": getattr(job, 'id', None)
            }
        except Exception as e:
                logger.exception("Failed to enqueue training job via RQ")
                # Persist failure status for the UI
                try:
                    os.makedirs(os.path.dirname(EVAL_FILE), exist_ok=True)
                    with open(EVAL_FILE, "w") as f:
                        json.dump({"status": "failed", "message": str(e)}, f)
                except Exception as ie:
                    logger.warning(f"Failed to write failed training status file {EVAL_FILE}: {ie}")

                raise HTTPException(status_code=503, detail="Training queue unavailable: " + str(e))

    except Exception as e:
        # Persist failure status for the UI
        try:
            os.makedirs(os.path.dirname(EVAL_FILE), exist_ok=True)
            with open(EVAL_FILE, "w") as f:
                json.dump({"status": "failed", "message": str(e)}, f)
        except Exception as ie:
            logger.warning(f"Failed to write failed training status file {EVAL_FILE}: {ie}")

        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run-cluster-background", tags=["Clustering"])
def run_cluster_background(payload: ClusterRequest):
    """Enqueue clustering as a background job and return the job id.

    Uses Redis/RQ similarly to the training endpoint.
    """
    try:
        from redis import Redis
        from rq import Queue

        redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        redis_conn = Redis.from_url(redis_url)
        q = Queue("clustering", connection=redis_conn)

        # Use the fully-qualified import path to avoid RQ resolution issues on package members.
        job = q.enqueue("utils.cluster_job.cluster_job", payload.object_name, job_timeout=3600)

        return {"status": "queued", "job_id": getattr(job, 'id', None)}
    except Exception as e:
        logger.exception("Failed to enqueue clustering job: %s", e)
        raise HTTPException(status_code=503, detail="Clustering queue unavailable: " + str(e))


@app.get("/cluster-status/{job_id}", tags=["Clustering"])
def cluster_status(job_id: str):
    """Query the Redis/RQ job status and return result if finished."""
    try:
        from redis import Redis
        from rq.job import Job
        from rq.exceptions import NoSuchJobError

        redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        redis_conn = Redis.from_url(redis_url)

        try:
            job = Job.fetch(job_id, connection=redis_conn)
        except NoSuchJobError:
            return {"job_id": job_id, "status": "not_found", "result": None}

        status = job.get_status()
        result = None
        if status == 'finished':
            result = job.result
        elif status == 'failed':
            result = {"error": str(job.exc_info)}

        return {"job_id": job_id, "status": status, "result": result}
    except Exception as e:
        logger.exception("Failed to fetch job status for %s: %s", job_id, e)
        raise HTTPException(status_code=503, detail=str(e))


# =========================================================
# Training status endpoint (RQ)
# =========================================================
from rq.job import Job as RQJob

@app.get("/train-status/{job_id}", tags=["Training"])
def train_status(job_id: str):
    """Check RQ job status for the given job id."""
    try:
        from redis import Redis
        from rq.exceptions import NoSuchJobError
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        redis_conn = Redis.from_url(redis_url)

        try:
            job = RQJob.fetch(job_id, connection=redis_conn)
        except NoSuchJobError:
            return {"job_id": job_id, "status": "queued", "is_finished": False, "is_failed": False, "result": None}
    except Exception as e:
        logger.exception("Failed to fetch job status")
        raise HTTPException(status_code=500, detail=str(e))

    status = job.get_status()
    result = None
    error = None
    if job.is_failed:
        try:
            redis_conn.eval(
                """
                if redis.call('GET', KEYS[1]) == ARGV[1] then
                    return redis.call('DEL', KEYS[1])
                end
                return 0
                """,
                1,
                "training_lock",
                job_id,
            )
        except Exception:
            logger.exception(
                "Could not release stale training lock for failed job %s",
                job_id,
            )
    try:
        result = job.result
    except Exception:
        result = None

    if job.is_failed and job.exc_info:
        error_lines = job.exc_info.strip().splitlines()
        error = error_lines[-1] if error_lines else "Training job failed; no error details were recorded."

    return {
        "job_id": job_id,
        "status": status,
        "is_finished": job.is_finished,
        "is_failed": job.is_failed,
        "result": result,
        "error": error,
    }

# =========================================================
# Prediction
# =========================================================
@app.get("/metadata", tags=["Prediction"])
def model_metadata():
    response = httpx.get(f"{PREDICTION_URL}/metadata", timeout=10)
    return safe_json_response(response)

@app.get("/models", tags=["Prediction"])
def list_models():
    response = httpx.get(f"{PREDICTION_URL}/models", timeout=10)
    return safe_json_response(response)

from typing import Optional
from pydantic import BaseModel

class ModelSelect(BaseModel):
    model_name: str


@app.post("/models/select", tags=["Prediction"])
async def select_model(model_name: Optional[str] = None, body: Optional[ModelSelect] = None):
    """Select a model. Accepts either query param `model_name` or JSON body {"model_name": "..."}."""
    # prefer JSON body
    name = body.model_name if body is not None else model_name

    if not name:
        raise HTTPException(status_code=422, detail="model_name is required as query param or JSON body")

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(
            f"{PREDICTION_URL}/models/select",
            json={"model_name": name}
        )
    return safe_json_response(response)


@app.delete("/models/{model_name}", tags=["Prediction"])
async def delete_trained_model(model_name: str):
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.delete(f"{PREDICTION_URL}/models/{model_name}")
    return safe_json_response(response)

# =========================================================
# Version Management
# =========================================================
@app.get("/versions", tags=["Training"])
def list_versions():
    """Get all model versions with classification and clustering scores."""
    try:
        from utils.model_registry import get_versions
        versions = get_versions()
        
        # Format for display
        version_list = []
        for version_name, metadata in versions.items():
            classification_scores = {}
            for clf_name, clf_data in metadata.get("classification_results", {}).items():
                test_metrics = clf_data.get("test_metrics", {})
                classification_scores[clf_name] = {
                    "accuracy": test_metrics.get("accuracy"),
                    "precision": test_metrics.get("precision"),
                    "recall": test_metrics.get("recall"),
                    "f1_score": test_metrics.get("f1_score")
                }
            
            version_list.append({
                "version": version_name,
                "timestamp": metadata.get("timestamp"),
                "best_classifier": metadata.get("best_classifier"),
                "classifiers": metadata.get("classifiers"),
                "clusterers": metadata.get("clusterers"),
                "classification_scores": classification_scores,
                "clustering_results": metadata.get("clustering_results", {})
            })
        
        return {
            "status": "success",
            "versions": sorted(version_list, key=lambda x: x["version"]),
            "total": len(version_list)
        }
    except Exception as e:
        logger.exception("Failed to fetch versions")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/versions/activate/{version_name}", tags=["Training"])
def activate_version(version_name: str):
    """Activate a specific model version."""
    try:
        from utils.model_registry import get_version_by_name, set_active_model
        
        version = get_version_by_name(version_name)
        if not version:
            raise HTTPException(status_code=404, detail=f"Version {version_name} not found")
        
        # Activate the best classifier from that version
        best_classifier = version.get("best_classifier")
        if best_classifier:
            set_active_model(best_classifier)
            return {
                "status": "success",
                "message": f"Activated {version_name} (best_classifier: {best_classifier})"
            }
        else:
            raise HTTPException(status_code=400, detail=f"Version {version_name} has no best_classifier")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to activate version")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", tags=["Prediction"])
def predict(payload: PredictRequest):
    logger.info(f"Prediction request received: {payload.data}")

    request_payload = {"data": payload.data}

    response = httpx.post(
        f"{PREDICTION_URL}/predict",
        json=request_payload,
        timeout=30
    )

    result = safe_json_response(response)

    # Enrich response for interpretability
    result.setdefault("algorithm", "Unknown")
    result["features"] = [
        {"name": k, "value": v} for k, v in payload.data.items()
    ]
    result["feature_count"] = len(payload.data)

    logger.info(f"Prediction response: {result}")
    return result

# =========================================================
# Evaluation
# =========================================================
@app.get("/evaluation-results", tags=["Evaluation"])
def evaluation_results():
    if os.path.exists(EVAL_FILE):
        with open(EVAL_FILE) as f:
            results = json.load(f)

        if results.get("status") == "running":
            try:
                from redis import Redis
                from rq import Queue
                from rq.registry import StartedJobRegistry

                redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
                redis_conn = Redis.from_url(redis_url)
                training_queue = Queue("training", connection=redis_conn)
                queued_jobs = training_queue.count
                started_jobs = StartedJobRegistry(
                    queue=training_queue
                ).get_job_ids()
            except Exception as exc:
                logger.exception("Could not verify training queue status")
                results["status"] = "unavailable"
                results["message"] = f"Could not verify whether training is active: {exc}"
            else:
                if not queued_jobs and not started_jobs:
                    results["status"] = "interrupted"
                    results["message"] = (
                        "No queued or running training job was found. "
                        "The previous run may have been interrupted."
                    )

        return results

    return {
        "status": "not_ready",
        "message": "Evaluation results not available yet"
    }
