import os
import io
import pandas as pd
from minio import Minio
from redis import Redis
from rq import Queue, Worker
import time
from redis import exceptions

# Environment variables
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET = os.getenv("MINIO_SECRET_KEY", "minioadmin")

# MinIO client
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS,
    secret_key=MINIO_SECRET,
    secure=False
)

# Redis connection with retry
for _ in range(10):
    try:
        redis_conn = Redis.from_url(REDIS_URL)
        redis_conn.ping()
        print("Connected to Redis")
        break
    except exceptions.ConnectionError:
        print("Redis not ready, retrying...")
        time.sleep(2)
else:
    raise RuntimeError("Cannot connect to Redis")

queue = Queue("preprocess", connection=redis_conn)
queue_training = Queue("training", connection=redis_conn)
queue_clustering = Queue("clustering", connection=redis_conn)

from utils.preprocess_job import preprocess_job

# Training and clustering job wrappers are provided by their modules and will be
# executed by RQ when queued. We register all queues so the Worker polls them.

import threading
import pathlib

DATA_DIR = os.getenv("DATA_DIR", "/data")


def _file_trigger_poller(poll_interval: int = 5):
    """Polls the mounted data directory for a 'train_request.json' file and runs training when found."""
    trigger_path = pathlib.Path(DATA_DIR) / "train_request.json"
    from utils.train_job import train_job

    print(f"File trigger poller started, watching {trigger_path}")
    while True:
        try:
            if trigger_path.exists():
                processing_path = trigger_path.with_suffix('.processing')
                try:
                    # Attempt to atomically claim the trigger file by renaming it.
                    trigger_path.rename(processing_path)
                    print("Train request file claimed; starting background training job via file trigger")
                    try:
                        train_job()
                    except Exception as e:
                        print(f"Training job failed: {e}")
                    finally:
                        try:
                            processing_path.unlink()
                        except Exception:
                            pass
                except FileNotFoundError:
                    # File was claimed by another process/thread
                    pass
                except Exception as e:
                    print(f"Error claiming trigger file: {e}")
        except Exception:
            pass
        time.sleep(poll_interval)


if __name__ == "__main__":
    # Start file trigger poller thread (for environments where RQ is not used)
    t = threading.Thread(target=_file_trigger_poller, daemon=True)
    t.start()

    print("Worker started, polling 'preprocess', 'training', and 'clustering' queues...")
    worker = Worker([queue, queue_training, queue_clustering], connection=redis_conn)
    worker.work()
