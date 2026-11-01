import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from redis import Redis
from rq import Queue
from utils.preprocess_job import preprocess_job  # Job function
from minio import Minio
from rq.job import Job

app = FastAPI(
    title="Preprocessing API",
    version="1.0.0",
    description="API for submitting CSV files for preprocessing and checking job status"
)

# ---------------- Environment ----------------
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET = "uploads"

# ---------------- Clients ----------------
redis_conn = Redis.from_url(REDIS_URL)
queue = Queue("preprocess", connection=redis_conn)
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS,
    secret_key=MINIO_SECRET,
    secure=False
)

# Ensure bucket exists
if not minio_client.bucket_exists(BUCKET):
    minio_client.make_bucket(BUCKET)

# ---------------- Models ----------------
class FileRequest(BaseModel):
    filename: str

# ---------------- Endpoints ----------------
@app.post("/submit", summary="Submit file for preprocessing", tags=["Preprocessing"])
def submit_file(request: FileRequest):
    """
    Submit a CSV file already uploaded to MinIO for preprocessing.
    Returns the job ID and current status.
    """
    # Check if file exists
    try:
        minio_client.stat_object(BUCKET, request.filename)
    except Exception:
        raise HTTPException(status_code=404, detail=f"File not found in MinIO: {request.filename}")
    
    # Enqueue job
    job = queue.enqueue(preprocess_job, request.filename)
    return {"job_id": getattr(job, 'id', None), "status": job.get_status(), "object_name": request.filename}


@app.get("/status/{job_id}", summary="Check preprocessing job status", tags=["Preprocessing"])
def check_status(job_id: str):
    """
    Check the status of a preprocessing job by job_id.
    """
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")

    response = {
        "job_id": job.id,
        "status": job.get_status(),
        "result": job.result,
        "failed": job.is_failed
    }

    # Optionally return cleaned file info if job finished
    if job.is_finished and job.result is None:
        response["cleaned_object_name"] = f"cleaned_{job.args[0]}"

    return response
