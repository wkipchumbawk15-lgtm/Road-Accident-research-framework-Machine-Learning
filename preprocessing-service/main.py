from fastapi import FastAPI, UploadFile, File, HTTPException
from datetime import timedelta
import io
import os
import redis
import rq
from rq.job import Job
from minio import Minio
from utils.preprocess_job import preprocess_job

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET = os.getenv("MINIO_BUCKET", "uploads")

redis_conn = redis.from_url(REDIS_URL)
queue = rq.Queue("preprocess", connection=redis_conn)
minio_client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=False)
if not minio_client.bucket_exists(BUCKET):
    minio_client.make_bucket(BUCKET)

app = FastAPI(title="Accident Data Preprocessing Service")

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="A dataset filename is required")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="Supported formats are CSV, XLSX and XLS")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    base, extension = os.path.splitext(file.filename)
    from datetime import datetime
    versioned = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{extension}"
    minio_client.put_object(BUCKET, versioned, io.BytesIO(raw), length=len(raw))
    job = queue.enqueue(preprocess_job, versioned)
    # RQ Job exposes `id` property for the job identifier
    return {"status": "queued", "job_id": job.id, "object_name": versioned}

@app.get("/status/{job_id}")
def job_status(job_id: str):
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")
    response = {"job_id": job.id, "status": job.get_status(), "result": job.result, "failed": job.is_failed}
    if job.is_finished and job.result:
        response["download_url"] = minio_client.presigned_get_object(BUCKET, job.result, expires=timedelta(minutes=60))
    if job.is_failed:
        response["error"] = str(job.exc_info or "preprocessing failed")
    return response
