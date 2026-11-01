import os
import io
import json
import logging
from datetime import datetime

import httpx
import pandas as pd
import math
import numpy as np

logger = logging.getLogger("cluster_job")

# Environment
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET = os.getenv("MINIO_BUCKET", "uploads")
CLUSTER_URL = os.getenv("CLUSTER_URL", "http://clustering:8000")


def _get_minio_client():
    try:
        from minio import Minio
        client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=False)
        return client
    except Exception as e:
        logger.exception("Failed to init MinIO client: %s", e)
        return None


def _find_candidate_key(client, object_name):
    if client is None or not object_name:
        return None

    # Prefer real data files, not result blobs or metadata JSON.
    normalized_obj = object_name.strip()
    candidates = [normalized_obj, f"cleaned_{normalized_obj}", normalized_obj.replace(" ", "%20")]
    for c in candidates:
        try:
            if c.lower().endswith((".csv", ".tsv", ".txt")):
                client.stat_object(BUCKET, c)
                return c
        except Exception:
            continue

    # fallback to prefix search
    base, _ = os.path.splitext(normalized_obj)
    prefix = f"cleaned_{base}"
    try:
        for obj in client.list_objects(BUCKET, recursive=True):
            key = obj.object_name
            if key.startswith(prefix) and key.lower().endswith((".csv", ".tsv", ".txt")):
                return key
    except Exception:
        pass

    # Last fallback: pick the most recent direct CSV-like object in the bucket.
    try:
        matches = []
        for obj in client.list_objects(BUCKET, recursive=True):
            key = obj.object_name
            if key.lower().endswith((".csv", ".tsv", ".txt")) and not key.startswith("clustering_"):
                matches.append(obj)
        if matches:
            latest = max(matches, key=lambda o: getattr(o, "last_modified", datetime.min))
            return latest.object_name
    except Exception:
        pass
    return None


def cluster_job(object_name: str):
    """Background job that runs clustering for a dataset stored in MinIO.

    Returns a dict with `clustering_key` (MinIO object key) and `size`.
    """
    client = _get_minio_client()
    if client is None:
        raise RuntimeError("MinIO unavailable")

    key = _find_candidate_key(client, object_name) or object_name
    try:
        o = client.get_object(BUCKET, key)
        data = o.read()
    except Exception as e:
        logger.exception("Failed to read object %s: %s", key, e)
        raise

    # Try to parse CSV into records. Some NTSA exports are not strict UTF-8 and
    # may also contain delimiter or row issues. Try several reasonable encodings
    # and delimiters while skipping obviously malformed lines.
    def _read_csv_fallback(payload: bytes):
        last_error = None
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            for sep in (",", ";", "\t", "|", None):
                try:
                    kwargs = {"encoding": encoding, "engine": "python", "on_bad_lines": "skip"}
                    if sep is not None:
                        kwargs["sep"] = sep
                    return pd.read_csv(io.BytesIO(payload), **kwargs)
                except (UnicodeDecodeError, ValueError, pd.errors.ParserError) as exc:
                    last_error = exc
                    continue
        # Final defensive fallback: decode with replacement and auto-detect delimiters.
        try:
            text = payload.decode("utf-8", errors="replace")
            return pd.read_csv(io.StringIO(text), sep=None, engine="python", on_bad_lines="skip")
        except Exception as exc:
            logger.exception("Unable to parse CSV payload for %s after trying common encodings", key)
            raise last_error or exc

    try:
        df = _read_csv_fallback(data)
        if df.empty:
            raise ValueError(f"CSV payload for {key} was empty after parsing")
        records = df.to_dict("records")
    except Exception as e:
        logger.exception("Failed to parse CSV for %s: %s", key, e)
        raise

    # Sanitize records for JSON (replace NaN/inf and convert numpy types)
    def _sanitize_value(v):
        # pandas NA
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass

        # numpy scalar types
        if isinstance(v, (np.floating,)):
            if math.isnan(v) or math.isinf(v):
                return None
            return float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, (list, tuple)):
            return [_sanitize_value(x) for x in v]
        # floats/ints
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        if isinstance(v, int):
            return v
        return v

    sanitized = []
    for rec in records:
        newrec = {k: _sanitize_value(v) for k, v in rec.items()}
        sanitized.append(newrec)

    # Post to clustering service
    try:
        resp = httpx.post(f"{CLUSTER_URL}/run-cluster", json=sanitized, timeout=600)
    except Exception as e:
        # If JSON encoding fails due to out-of-range floats, attempt a best-effort retry
        logger.exception("HTTP request to clustering service failed: %s", e)
        try:
            # Final defensive pass: stringify remaining problematic values
            def _stringify_value(v):
                try:
                    if v is None:
                        return None
                    if isinstance(v, (str, int, float, bool)):
                        return v
                    return str(v)
                except Exception:
                    return None

            sanitized2 = []
            for rec in sanitized:
                sanitized2.append({k: _stringify_value(v) for k, v in rec.items()})

            resp = httpx.post(f"{CLUSTER_URL}/run-cluster", json=sanitized2, timeout=600)
        except Exception:
            logger.exception("Retry to clustering service failed after sanitization")
            raise

    if resp.status_code >= 400:
        logger.error("Clustering service returned error: %s %s", resp.status_code, resp.text)
        raise RuntimeError(f"Clustering service error: {resp.status_code}")

    content = resp.content
    size = len(content)
    SIZE_THRESHOLD = int(os.getenv('CLUSTER_RESPONSE_SIZE_THRESHOLD', 1_000_000))

    base, _ = os.path.splitext(object_name)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    clustering_key = f"clustering_{base}_{ts}.json"

    try:
        client.put_object(BUCKET, clustering_key, io.BytesIO(content), length=size, content_type='application/json')
        logger.info("Persisted clustering result %s (%d bytes)", clustering_key, size)
    except Exception as e:
        logger.exception("Failed to persist clustering result to MinIO: %s", e)
        raise

    return {"clustering_key": clustering_key, "size": size}
