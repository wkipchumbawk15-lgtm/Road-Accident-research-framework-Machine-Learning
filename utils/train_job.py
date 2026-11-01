import io
import json
import os
from pathlib import Path
import uuid
from redis import Redis
from rq import get_current_job
from minio import Minio
from utils.train_model import train_all_models
from utils.model_registry import create_version
from utils.clustering_engine import run_clustering
from utils.research_schema import normalize_column_name
import pandas as pd

EVAL_FILE = Path("/data/evaluation_results.json")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
LOCK_KEY = "training_lock"
LOCK_TTL = 60 * 60 * 2
BUCKET = os.getenv("MINIO_BUCKET", "uploads")


def _download_cleaned(object_name: str) -> str:
    """Download the processed MinIO dataset to the shared /data volume."""
    client = Minio(
        os.getenv("MINIO_ENDPOINT", "minio:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=False,
    )
    obj = client.get_object(BUCKET, object_name)
    raw = obj.read()
    path = Path("/data") / Path(object_name).name
    path.write_bytes(raw)
    return str(path)


def train_job(dataset_object=None, target_column=None, classifiers=None, clusterers=None):
    classifiers = classifiers or ["random_forest", "svm"]
    clusterers = clusterers or ["kmeans", "dbscan", "agglomerative", "mean_shift"]
    if not dataset_object:
        raise ValueError("dataset_object is required for research training; upload and preprocess the NTSA dataset first.")
    if not target_column:
        raise ValueError("target_column is required; the NTSA dataset has no built-in accident_severity field.")

    r = Redis.from_url(REDIS_URL)
    current_job = get_current_job(connection=r)
    lock_token = current_job.id if current_job else str(uuid.uuid4())
    if not r.set(LOCK_KEY, lock_token, nx=True, ex=LOCK_TTL):
        return {"status": "skipped", "message": "Another training run is in progress"}

    try:
        EVAL_FILE.parent.mkdir(parents=True, exist_ok=True)
        EVAL_FILE.write_text(json.dumps({"status": "running", "message": "Research workflow in progress"}))
        target = normalize_column_name(target_column)
        if target != "cause_category":
            raise ValueError(
                "The proposal-aligned classification target is 'cause_category', which groups "
                "accidents by mapped cause. DATE and other fields are not approved targets for this workflow."
            )

        dataset_path = _download_cleaned(dataset_object)
        df = pd.read_csv(dataset_path)
        if target not in {normalize_column_name(column) for column in df.columns}:
            raise ValueError(
                "The processed dataset has no 'cause_category'. Check the cause-code mapping "
                "and preprocess the dataset again before training."
            )

        info = train_all_models(dataset_path=dataset_path, target_column=target_column, classifiers=classifiers)
        clustering_results = run_clustering(df, algorithms=clusterers)

        version_name = create_version(
            classifiers=classifiers,
            clusterers=clusterers,
            classification_results=info["results"],
            clustering_results=clustering_results,
            dataset_object=dataset_object,
            target_column=target_column,
            preprocessing={"cleaned_dataset": dataset_object},
            feature_selection={"method": "SelectPercentile(f_classif)", "percentile": 80},
        )
        payload = {
            "status": "ready",
            "active_model": info.get("best_model"),
            "version": version_name,
            "dataset_object": dataset_object,
            "target_column": info.get("target_column"),
            "predictors": info.get("predictors"),
            "classification": info.get("results"),
            "evaluation": info.get("results"),
            "clustering": clustering_results,
        }
        EVAL_FILE.write_text(json.dumps(payload, indent=2, default=str))
        return payload
    except Exception as exc:
        EVAL_FILE.write_text(json.dumps({"status": "failed", "message": str(exc)}, indent=2))
        raise
    finally:
        r.eval(
            """
            if redis.call('GET', KEYS[1]) == ARGV[1] then
                return redis.call('DEL', KEYS[1])
            end
            return 0
            """,
            1,
            LOCK_KEY,
            lock_token,
        )
