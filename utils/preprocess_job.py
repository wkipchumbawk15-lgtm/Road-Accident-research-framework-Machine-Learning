"""Shared NTSA preprocessing job used by the API service and RQ worker."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import pandas as pd
from minio import Minio
from .research_schema import engineer_ntsa_features
from .workbook import _combine_excel_sheets

BUCKET = os.getenv("MINIO_BUCKET", "uploads")


def _client():
    return Minio(
        os.getenv("MINIO_ENDPOINT", "minio:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=False,
    )


def _read_table(raw: bytes, object_name: str) -> pd.DataFrame:
    ext = Path(object_name).suffix.lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(raw), sheet_name=None)
    return pd.read_csv(io.BytesIO(raw))


def preprocess_job(object_name: str):
    client = _client()
    obj = client.get_object(BUCKET, object_name)
    raw = obj.read()
    ext = Path(object_name).suffix.lower()

    if ext in {".xlsx", ".xls"}:
        df, cause_code_mapping = _combine_excel_sheets(raw)
    else:
        df = pd.read_csv(io.BytesIO(raw))
        cause_code_mapping = None

    rows_before = len(df)
    original_columns = list(df.columns)
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    df = engineer_ntsa_features(df, cause_code_mapping=cause_code_mapping)

    # Keep rows with missing date/time rather than inventing values. The missingness
    # indicators make data-quality limitations visible to the research workflow.
    quality = {
        "missing_counts_before": {str(k): int(v) for k, v in pd.DataFrame(pd.read_csv(io.BytesIO(raw))).isna().sum().items()} if ext not in {".xlsx", ".xls"} else {},
        "missing_counts_after": {str(k): int(v) for k, v in df.isna().sum().items()},
        "duplicate_rows_removed": duplicate_count,
        "rows_before": rows_before,
        "rows_after": len(df),
    }

    cleaned_name = f"cleaned_{object_name.rsplit('.', 1)[0]}.csv"
    out = df.to_csv(index=False).encode("utf-8")
    client.put_object(BUCKET, cleaned_name, io.BytesIO(out), length=len(out), content_type="text/csv")

    metadata = {
        "dataset_type": "NTSA road accident research dataset",
        "source_object": object_name,
        "cleaned_object": cleaned_name,
        "original_columns": original_columns,
        "engineered_columns": [c for c in df.columns if c not in original_columns],
        "cause_code_mapping_source": (
            "embedded workbook mapping sheet" if cause_code_mapping is not None
            else "project CSV mapping"
        ),
        "cause_category_values": sorted(
            str(value) for value in df["cause_category"].dropna().unique()
        ) if "cause_category" in df.columns else [],
        "quality": quality,
        "research_notes": [
            "Missing DATE values are retained and flagged; dates are not forward-filled automatically.",
            "Age values that are non-numeric or outside 0-100 are treated as missing.",
            "NTSA fields are not converted into unsupported weather, road-condition, speed, latitude or longitude features.",
        ],
    }
    metadata_name = f"metadata_{object_name.rsplit('.', 1)[0]}.json"
    meta = json.dumps(metadata, indent=2).encode("utf-8")
    client.put_object(BUCKET, metadata_name, io.BytesIO(meta), length=len(meta), content_type="application/json")
    return cleaned_name
