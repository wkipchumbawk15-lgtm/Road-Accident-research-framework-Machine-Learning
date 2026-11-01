"""Feature engineering and feature-selection helpers for the NTSA research dataset."""
from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "year", "month", "hour", "minute", "day_of_week", "is_weekend",
    "age", "no", "date_missing", "time_missing", "age_missing"
]
CATEGORICAL_FEATURES = [
    "county", "vehicle_type", "gender", "victim", "age_group", "time_period"
]
HIGH_CARDINALITY_FEATURES = ["road", "place"]


def recover_group_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Recover continuation dates used by the NTSA workbook without fabricating dates.

    In the supplied workbook, DATE is populated on the first record of a date group
    and is blank on many continuation rows. A recovered date is accepted only when
    it agrees with the recorded DAY field (or DAY is missing). Otherwise the date
    remains missing and the inconsistency is flagged.
    """
    df = df.copy()
    if "date" not in df.columns:
        return df

    explicit = pd.to_datetime(df["date"], errors="coerce")
    df["date_explicit"] = explicit.notna().astype(int)
    if "source_sheet" in df.columns:
        group = df["source_sheet"].astype(str)
    else:
        group = pd.Series("dataset", index=df.index)

    inherited = explicit.groupby(group).ffill()
    day_map = {name: i for i, name in enumerate(
        ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
    )}
    recorded_day = df.get("day", pd.Series(pd.NA, index=df.index)).astype("string").str.strip().str.upper().map(day_map)
    inherited_day = inherited.dt.dayofweek
    compatible = inherited.notna() & (recorded_day.isna() | (recorded_day == inherited_day))

    df["analysis_date"] = explicit.where(explicit.notna(), inherited.where(compatible))
    df["date_recovered"] = (explicit.isna() & df["analysis_date"].notna()).astype(int)
    df["date_day_conflict"] = (inherited.notna() & recorded_day.notna() & (recorded_day != inherited_day)).astype(int)
    return df


def make_research_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create analysis features from the supplied NTSA variables."""
    df = df.copy()
    df = recover_group_dates(df)

    # Ensure `analysis_date` is a Series even if the column is missing
    if "analysis_date" in df.columns:
        analysis_date = pd.to_datetime(df["analysis_date"], errors="coerce")
    else:
        analysis_date = pd.Series(pd.NaT, index=df.index)

    if "source_sheet" in df.columns:
        source_year = pd.to_numeric(df["source_sheet"], errors="coerce")
    else:
        source_year = pd.Series(np.nan, index=df.index)

    df["year"] = analysis_date.dt.year.fillna(source_year).astype("Int64")
    df["month"] = analysis_date.dt.month.astype("Int64")

    day_map = {name: i for i, name in enumerate(
        ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
    )}
    recorded_day = df.get("day", pd.Series(pd.NA, index=df.index)).astype("string").str.strip().str.upper().map(day_map)
    df["day_of_week"] = analysis_date.dt.dayofweek.astype("Int64").fillna(recorded_day.astype("Int64"))
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    if "time_24_hours" in df.columns:
        raw = df["time_24_hours"]
        numeric = pd.to_numeric(raw, errors="coerce")
        valid = numeric.between(0, 2359)
        padded = numeric.where(valid).round().astype("Int64").astype("string").str.zfill(4)
        df["hour"] = pd.to_numeric(padded.str[:2], errors="coerce").astype("Int64")
        df["minute"] = pd.to_numeric(padded.str[2:], errors="coerce").astype("Int64")
        df["time_missing"] = df["hour"].isna().astype(int)
        df["time_period"] = pd.cut(
            pd.to_numeric(df["hour"], errors="coerce"),
            bins=[-1, 5, 11, 17, 23],
            labels=["NIGHT", "MORNING", "AFTERNOON", "EVENING"],
        ).astype("string")

    if "age" in df.columns:
        raw_age = pd.to_numeric(df["age"], errors="coerce")
        invalid_age = raw_age.notna() & ~raw_age.between(0, 100)
        df["age"] = raw_age.mask(invalid_age)
        df["age_missing"] = df["age"].isna().astype(int)
        df["age_group"] = pd.cut(
            df["age"], bins=[-1, 17, 35, 55, 100],
            labels=["MINOR", "YOUTH", "ADULT", "SENIOR"]
        ).astype("string")

    for col in ["county", "road", "place", "vehicle_type", "gender", "victim"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().str.upper()
            df[col] = df[col].replace({"NAN": pd.NA, "NONE": pd.NA, "": pd.NA})

    if "vehicle_type" in df.columns:
        df["vehicle_type"] = df["vehicle_type"].replace({
            "MOTOR CYCLE": "MOTORCYCLE",
            "MOTOR BIKE": "MOTORCYCLE",
            "PSV BUS": "PSV",
        })

    if "gender" in df.columns:
        df["gender"] = df["gender"].replace({
            "M ": "M", "M&F": "M & F", "M/F": "M & F", "F&M": "M & F",
        })

    if "victim" in df.columns:
        df["victim"] = df["victim"].replace({
            "PEDESTRIAN": "PED", "PEDESTRAIN": "PED", "PED ": "PED",
            "M/CYCLIST": "MCY", "M/CY": "MCY",
        })

    if "no" in df.columns:
        df["no"] = pd.to_numeric(df["no"], errors="coerce")
    # Fill categorical missing values with a sentinel to ensure encoders
    # and downstream clustering steps don't receive null categories.
    categorical_fill_cols = [c for c in (CATEGORICAL_FEATURES + HIGH_CARDINALITY_FEATURES) if c in df.columns]
    for col in categorical_fill_cols:
        # Preserve string dtype and use a clear sentinel value
        df[col] = df[col].astype("string").fillna("MISSING")

    return df



def harmonize_ntsa_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Create audit-preserving canonical categories for NTSA clustering.

    Raw source fields are retained. Harmonized fields reduce spelling, separator,
    ordering and abbreviation variation without overwriting the source values.
    """
    import re
    from rapidfuzz import process, fuzz
    out = df.copy()
    counties = ["BARINGO","BOMET","BUNGOMA","BUSIA","ELGEYO-MARAKWET","EMBU","GARISSA","HOMA BAY","ISIOLO","KAJIADO","KAKAMEGA","KERICHO","KIAMBU","KILIFI","KIRINYAGA","KISII","KISUMU","KITUI","KWALE","LAIKIPIA","LAMU","MACHAKOS","MAKUENI","MANDERA","MARSABIT","MERU","MIGORI","MOMBASA","MURANGA","NAIROBI","NAKURU","NANDI","NAROK","NYAMIRA","NYANDARUA","NYERI","SAMBURU","SIAYA","TAITA-TAVETA","TANA RIVER","THARAKA-NITHI","TRANS NZOIA","TURKANA","UASIN GISHU","VIHIGA","WAJIR","WEST POKOT"]
    def basic(x):
        if pd.isna(x): return None
        return re.sub(r"\s+", " ", str(x).upper().strip().replace("’", "'"))
    county_map = {"NRBI":"NAIROBI","NAIRIBI":"NAIROBI","MAIROBI":"NAIROBI","NAIROBIB":"NAIROBI","KAIMBU":"KIAMBU","NROK":"NAROK","MACHAK0S":"MACHAKOS","MAUENI":"MAKUENI","KAKEMEGA":"KAKAMEGA","KIRINTAGA":"KIRINYAGA","KIRINYIGA":"KIRINYAGA","NHYANDARUA":"NYANDARUA","ENBU":"EMBU","ELEGYO MARAKWET":"ELGEYO-MARAKWET","ELEGEYO MARAKWET":"ELGEYO-MARAKWET","ELGEIYO MARAKWET":"ELGEYO-MARAKWET","UASIN-GISHU":"UASIN GISHU","THARAK NITHI":"THARAKA-NITHI","W.POKOT":"WEST POKOT","TAITA":"TAITA-TAVETA","MURANG;A":"MURANGA","MURANG’A":"MURANGA","KISLL":"KISII","BERINGO":"BARINGO","SOTIK":"KERICHO","MWINGI":"KITUI","MUMIA":"KAKAMEGA","MWEA":"KIRINYAGA","MALINDI":"KILIFI","KISERIAN":"KAJIADO","NYAHURURU":"LAIKIPIA","LODWAR":"TURKANA"}
    def county(x):
        s=basic(x)
        if not s: return "MISSING"
        s=s.replace(" COUNTY","").replace("_"," ").replace(";"," ")
        if s in county_map: return county_map[s]
        if s in counties: return s
        m=process.extractOne(s,counties,scorer=fuzz.ratio)
        return m[0] if m and m[1] >= 90 else s
    def vehicle(x):
        s=basic(x)
        if not s: return "MISSING"
        s=s.replace("&","|").replace(",","|").replace(" AND ","|")
        s=re.sub(r"\s*[/+]\s*","|",s)
        aliases={"MOTOR CYCLE":"MOTORCYCLE","MOTOR BIKE":"MOTORCYCLE","M/CYCLE":"MOTORCYCLE","M/CY":"MOTORCYCLE","M CYCLE":"MOTORCYCLE","TRI CYCLE":"TRICYCLE","TRI-CYCLE":"TRICYCLE","TRI CYCLES":"TRICYCLE","P/CYCLE":"PEDALCYCLE","P/CYCLIST":"PEDALCYCLE","PEDAL CYCLE":"PEDALCYCLE","PEDAL CYCLIST":"PEDALCYCLE","UNKNOWN VEHICLE":"UNKNOWN","UNKNOWN M/V":"UNKNOWN","UNKNOWN MV":"UNKNOWN","M/V":"UNKNOWN","PRVATE":"PRIVATE","COMERCIAL":"COMMERCIAL","HAND CART":"HANDCART"}
        vals=[]
        for t in s.split("|"):
            t=re.sub(r"\s+"," ",t.strip())
            if t: vals.append(aliases.get(t,t))
        return "|".join(sorted(set(vals))) if vals else "UNKNOWN"
    def victim(x):
        s=basic(x)
        if not s: return "MISSING"
        s=s.replace("PEDESTRAIN","PEDESTRIAN").replace("PEDD","PED").replace("PE D","PED")
        if "PED" in s or "PEDESTRIAN" in s or "HANDCART" in s: return "PEDESTRIAN"
        if "MCY" in s or "M/CY" in s or "M/CYCLIST" in s: return "MOTORCYCLIST"
        if "PCY" in s or "P/CY" in s or "CYCLIST" in s: return "CYCLIST"
        if "PIL" in s or "PILLION" in s: return "PILLION"
        if "PAS" in s or "PASS" in s: return "PASSENGER"
        if "DRV" in s or "DRIVER" in s or s=="DR": return "DRIVER"
        return "OTHER"
    def gender(x):
        s=basic(x)
        if not s or s in {"-","_","0","N"}: return "UNKNOWN"
        m=bool(re.search(r"(?<![A-Z])M(?![A-Z])|\b\d+M\b|^M\d",s)); f=bool(re.search(r"(?<![A-Z])F(?![A-Z])|\b\d+F\b|^F\d",s)); j=bool(re.search(r"JUV|\bJ\b|\bA\b",s))
        if m and f: return "MIXED_M_F"
        if m and j: return "MALE_JUVENILE"
        if f and j: return "FEMALE_JUVENILE"
        if j: return "JUVENILE"
        if m: return "MALE"
        if f: return "FEMALE"
        return "UNKNOWN"
    if "county" in out: out["county_h"]=out["county"].map(county)
    if "vehicle_type" in out: out["vehicle_type_h"]=out["vehicle_type"].map(vehicle)
    if "victim" in out: out["victim_h"]=out["victim"].map(victim)
    if "gender" in out: out["gender_h"]=out["gender"].map(gender)
    return out

def build_cluster_preprocessor(df: pd.DataFrame, include_high_cardinality: bool = False):
    """Build a reproducible mixed-type transformer for clustering."""
    # Exclude features that are not present or entirely null to avoid
    # downstream failures in imputers or encoders (e.g., OneHotEncoder
    # cannot operate on categorical features with all-null values).
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns and not df[c].isna().all()]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns and not df[c].isna().all()]
    if include_high_cardinality:
        categorical += [c for c in HIGH_CARDINALITY_FEATURES if c in df.columns and not df[c].isna().all()]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ], remainder="drop"), numeric, categorical


def select_supervised_features(X: pd.DataFrame, y: pd.Series, percentile: int = 80):
    """Fit the proposal's supervised feature-selection stage.

    This must only be used for classification and must never include the target
    itself among predictors.
    """
    preprocessor, numeric, categorical = build_cluster_preprocessor(X)
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("select", SelectPercentile(score_func=f_classif, percentile=percentile)),
    ])
    Xt = pipe.fit_transform(X, y)
    return pipe, Xt, {"method": "SelectPercentile(f_classif)", "percentile": percentile,
                       "numeric_features": numeric, "categorical_features": categorical}
