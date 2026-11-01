import os
from redis import Redis
from rq import Queue
from main import preprocess_job  # Import the function directly

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
redis_conn = Redis.from_url(REDIS_URL)
queue = Queue("preprocess", connection=redis_conn)

# Enqueue a file to preprocess
object_name = "accident_test_data.csv"
job = queue.enqueue(preprocess_job, object_name)
print(f"Enqueued job {job.id} for {object_name}")
