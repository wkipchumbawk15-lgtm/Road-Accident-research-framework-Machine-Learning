"""Research schema and feature engineering rules for the NTSA accident framework."""
from __future__ import annotations

from typing import Dict, List
from pathlib import Path
import re
import pandas as pd

NTSA_COLUMNS = [
    "date", "day", "time_24_hours", "county", "road", "place",
    "vehicle_type", "gender", "age", "cause_code", "victim", "no"
]

# Fields that describe the accident and may be used as predictors.
BASE_PREDICTORS = [
    "year", "month", "hour", "minute", "day_of_week", "is_weekend",
    "county", "road", "place", "vehicle_type", "gender", "age",
    "age_group", "victim", "time_period", "no"
]

TIME_ALIASES = {
    "time_24_hours": "time_24_hours",
    "time_24": "time_24_hours",
    "time_24hrs": "time_24_hours",
    "time": "time_24_hours",
}

COLUMN_ALIASES = {
    "date": "date",
    "day": "day",
    "time_24_hours": "time_24_hours",
    "time_24_hours_": "time_24_hours",
    "county": "county",
    "road": "road",
    "place": "place",
    "type_of_vehicle": "vehicle_type",
    "vehicle_type": "vehicle_type",
    "gender": "gender",
    "age": "age",
    "cause_code": "cause_code",
    "victim": "victim",
    "no": "no",
    "no.": "no",
}


def normalize_column_name(value: str) -> str:
    value = str(value).strip().lower()
    value = value.replace("/", "_").replace("-", "_")
    value = re.sub(r"[^a-z0-9_.]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]
    rename = {c: COLUMN_ALIASES[c] for c in df.columns if c in COLUMN_ALIASES}
    return df.rename(columns=rename)


def parse_ntsa_time(value):
    """Convert NTSA HHMM-like values (e.g. 600, 45, 2300) to hour/minute."""
    if pd.isna(value):
        return pd.NA, pd.NA
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "a", "-", "."}:
        return pd.NA, pd.NA
    try:
        number = int(float(text))
    except (ValueError, TypeError):
        return pd.NA, pd.NA
    if number < 0 or number > 2359:
        return pd.NA, pd.NA
    padded = f"{number:04d}"
    hour, minute = int(padded[:2]), int(padded[2:])
    if hour > 23 or minute > 59:
        return pd.NA, pd.NA
    return hour, minute


def engineer_ntsa_features(
    df: pd.DataFrame,
    cause_code_mapping: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply the canonical NTSA feature-engineering pipeline."""
    from utils.feature_engineering import make_research_features
    result = make_research_features(normalize_columns(df))

    mapping_path = Path(__file__).with_name("cause_code_mapping.csv")
    if cause_code_mapping is None and "cause_category" not in result.columns and mapping_path.exists():
        cause_code_mapping = pd.read_csv(mapping_path, dtype={"cause_code": "string"})

    if "cause_code" in result.columns and cause_code_mapping is not None:
        mapping = cause_code_mapping.copy()
        mapping.columns = [normalize_column_name(column) for column in mapping.columns]
        mapping = mapping.rename(columns={
            "code": "cause_code",
            "description": "cause_description",
            "category": "cause_category",
        })
        required_mapping_columns = {"cause_code", "cause_description"}
        missing_columns = required_mapping_columns - set(mapping.columns)
        if missing_columns:
            raise ValueError(
                f"Cause-code mapping is missing required columns: {sorted(missing_columns)}"
            )
        mapping["cause_code"] = mapping["cause_code"].astype("string").str.strip().str.upper()
        mapping = mapping.dropna(subset=["cause_code"]).drop_duplicates()
        if mapping["cause_code"].duplicated().any():
            raise ValueError("Cause-code mapping contains conflicting entries for a code.")
        mapping_columns = ["cause_code", "cause_description"]
        if "cause_category" in mapping.columns:
            mapping_columns.append("cause_category")
        if "cause_actor" in mapping.columns:
            mapping_columns.append("cause_actor")
        if "mapping_source" in mapping.columns:
            mapping_columns.append("mapping_source")
        else:
            mapping["mapping_source"] = "uploaded workbook mapping"
            mapping_columns.append("mapping_source")

        result["cause_code"] = result["cause_code"].astype("string").str.strip().str.upper()
        result = result.drop(
            columns=["cause_description", "cause_actor", "mapping_source", "cause_category"],
            errors="ignore",
        )
        result = result.merge(
            mapping[mapping_columns],
            on="cause_code",
            how="left",
            validate="many_to_one",
        )
        category_mapping_path = Path(__file__).with_name("cause_category_mapping.csv")
        if "cause_category" not in mapping.columns and category_mapping_path.exists():
            category_mapping = pd.read_csv(category_mapping_path)
            result = result.merge(
                category_mapping[["cause_description", "cause_category"]],
                on="cause_description",
                how="left",
                validate="many_to_one",
            )
    return result

def candidate_predictors(df: pd.DataFrame, target_column: str | None = None) -> List[str]:
    target_column = normalize_column_name(target_column) if target_column else None
    return [c for c in BASE_PREDICTORS if c in df.columns and c != target_column]
