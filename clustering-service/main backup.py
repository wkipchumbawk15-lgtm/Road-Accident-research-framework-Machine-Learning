# from fastapi import FastAPI, HTTPException
# from minio import Minio
# import io, pandas as pd
# from sklearn.cluster import DBSCAN
# import os

# MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
# MINIO_ACCESS = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
# MINIO_SECRET = os.getenv("MINIO_SECRET_KEY", "minioadmin")

# minio_client = Minio(
#     MINIO_ENDPOINT,
#     access_key=MINIO_ACCESS,
#     secret_key=MINIO_SECRET,
#     secure=False
# )

# app = FastAPI(title="Clustering Service")
# BUCKET = "uploads"

# @app.post("/run-cluster")
# def run_cluster(payload: dict):
#     try:
#         object_name = payload.get("object_name")
#         if not object_name:
#             raise HTTPException(status_code=400, detail="provide object_name")

#         cleaned_name = f"cleaned_{object_name}"

#         # 1. Get object
#         obj = minio_client.get_object(BUCKET, cleaned_name)

#         # 2. Read CSV
#         df = pd.read_csv(io.BytesIO(obj.read()))

#         # 3. Validate columns
#         if not {"lat", "lon"}.issubset(df.columns):
#             raise HTTPException(
#                 status_code=400,
#                 detail="no lat/lon columns in cleaned file"
#             )

#         coords = df[["lat", "lon"]].dropna().values
#         if len(coords) == 0:
#             raise HTTPException(status_code=400, detail="no coords")

#         # 4. Run clustering
#         model = DBSCAN(eps=0.01, min_samples=10).fit(coords)
#         labels = model.labels_

#         df.loc[df[["lat","lon"]].dropna().index, "cluster"] = labels

#         # 5. Save result
#         out = df.to_csv(index=False).encode()
#         minio_client.put_object(
#             BUCKET,
#             f"clustered_{object_name}",
#             io.BytesIO(out),
#             len(out),
#             content_type="text/csv"
#         )

#         n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

#         return {
#             "status": "done",
#             "clusters": int(n_clusters),
#             "output": f"clustered_{object_name}"
#         }

#     except HTTPException:
#         raise

#     except Exception as e:
#         # IMPORTANT: return JSON even on failure
#         raise HTTPException(status_code=500, detail=str(e))


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from minio import Minio
import io
import pandas as pd
from sklearn.cluster import DBSCAN
import os
import logging

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- ENV ----------------
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET = "uploads"

# ---------------- CLIENT ----------------
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS,
    secret_key=MINIO_SECRET,
    secure=False
)

# ---------------- APP ----------------
app = FastAPI(
    title="Clustering Service",
    version="1.0.0",
    description="Performs DBSCAN clustering on cleaned accident datasets"
)

# ---------------- MODELS ----------------
class ClusterRequest(BaseModel):
    object_name: str

# ---------------- ENDPOINT ----------------
@app.post("/run-cluster", tags=["Clustering"])
def run_cluster(payload: ClusterRequest):
    try:
        object_name = payload.object_name
        cleaned_name = f"cleaned_{object_name}"

        logger.info("Starting clustering for file: %s", cleaned_name)

        # 1️⃣ Load cleaned file from MinIO
        try:
            obj = minio_client.get_object(BUCKET, cleaned_name)
        except Exception:
            logger.error("File not found in MinIO: %s", cleaned_name)
            raise HTTPException(
                status_code=404,
                detail=f"Cleaned file not found: {cleaned_name}"
            )

        df = pd.read_csv(io.BytesIO(obj.read()))

        # 2️⃣ Log RAW columns
        logger.info("RAW columns from CSV: %s", list(df.columns))

        # 3️⃣ Normalize column names
        df.columns = df.columns.str.strip().str.lower()

        # 4️⃣ Log NORMALIZED columns
        logger.info("NORMALIZED columns: %s", list(df.columns))

        # 5️⃣ Rename geo columns
        df = df.rename(columns={
            "latitude": "lat",
            "longitude": "lon",
            "lng": "lon",
            "long": "lon"
        })

        # 6️⃣ Log CLEANED columns
        logger.info("CLEANED columns after rename: %s", list(df.columns))

        # 7️⃣ Validate required columns
        if not {"lat", "lon"}.issubset(df.columns):
            logger.error(
                "Missing required columns. Found=%s",
                list(df.columns)
            )
            raise HTTPException(
                status_code=400,
                detail="Cleaned file must contain lat and lon columns"
            )

        coords = df[["lat", "lon"]].dropna()

        if coords.empty:
            logger.error("No valid lat/lon values after dropna")
            raise HTTPException(
                status_code=400,
                detail="No valid coordinates found for clustering"
            )

        # 8️⃣ Run DBSCAN
        model = DBSCAN(eps=0.01, min_samples=10)
        labels = model.fit_predict(coords.values)

        df.loc[coords.index, "cluster"] = labels

        # 9️⃣ Save clustered output
        output_name = f"clustered_{object_name}"
        out = df.to_csv(index=False).encode()

        minio_client.put_object(
            BUCKET,
            output_name,
            io.BytesIO(out),
            length=len(out),
            content_type="text/csv"
        )

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        logger.info(
            "Clustering completed: clusters=%s output=%s",
            n_clusters,
            output_name
        )

        return {
            "status": "done",
            "clusters": n_clusters,
            "input": cleaned_name,
            "output": output_name
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Unexpected clustering error")
        raise HTTPException(status_code=500, detail=str(e))
