"""Cluster profiling and latent-pattern analysis for the NTSA research framework."""
from __future__ import annotations
import pandas as pd
import numpy as np

PROFILE_FIELDS = ["county", "vehicle_type", "victim", "gender", "time_period"]
NUMERIC_FIELDS = ["year", "month", "day_of_week", "hour", "minute", "age", "no"]

def attach_kmeans_assignments(df, assignments):
    out = df.copy()
    labels = np.asarray(assignments)
    if len(labels) != len(out):
        raise ValueError("Assignment length does not match dataset rows")
    out["cluster"] = labels
    return out

def _top_values(s, n=5):
    x = s.fillna("MISSING").astype(str).str.strip()
    vc = x.value_counts()
    return "; ".join([f"{idx} ({val})" for idx, val in vc.head(n).items()])

def cluster_summary(df):
    rows=[]
    for cluster, g in df.groupby("cluster", sort=True):
        row={"cluster": int(cluster), "records": len(g), "percent": round(len(g)/len(df)*100,2)}
        for f in NUMERIC_FIELDS:
            row[f"{f}_median"] = g[f].median() if f in g else np.nan
            row[f"{f}_mean"] = g[f].mean() if f in g else np.nan
        for f in PROFILE_FIELDS:
            row[f"{f}_top5"] = _top_values(g[f]) if f in g else ""
        if "cause_code" in g:
            row["cause_code_top5"] = _top_values(g["cause_code"])
        rows.append(row)
    return pd.DataFrame(rows)

def categorical_profile(df, field):
    overall = df[field].fillna("MISSING").astype(str).str.strip().value_counts(normalize=True)
    rows=[]
    for cluster,g in df.groupby("cluster",sort=True):
        shares=g[field].fillna("MISSING").astype(str).str.strip().value_counts(normalize=True)
        for value,share in shares.head(20).items():
            rows.append({"cluster":int(cluster),"field":field,"category":value,
                         "cluster_count":int((g[field].fillna("MISSING").astype(str).str.strip()==value).sum()),
                         "cluster_share_pct":round(share*100,3),
                         "overall_share_pct":round(overall.get(value,0)*100,3),
                         "enrichment_ratio":round(share/overall[value],3) if value in overall and overall[value]>0 else np.nan})
    return pd.DataFrame(rows)

def cause_profile(df):
    cols=[c for c in ["cause_code","cause_description","cause_actor","mapping_source"] if c in df.columns]
    x=df[cols].copy()
    if "cause_code" not in x: return pd.DataFrame()
    rows=[]
    for cluster,g in df.groupby("cluster",sort=True):
        vc=g["cause_code"].fillna("MISSING").astype(str).str.strip().value_counts().head(15)
        for code,n in vc.items():
            desc=""
            actor=""
            if "cause_description" in g.columns:
                vals=g.loc[g["cause_code"].fillna("MISSING").astype(str).str.strip()==code,"cause_description"].dropna().astype(str)
                desc=vals.iloc[0] if len(vals) else ""
            if "cause_actor" in g.columns:
                vals=g.loc[g["cause_code"].fillna("MISSING").astype(str).str.strip()==code,"cause_actor"].dropna().astype(str)
                actor=vals.iloc[0] if len(vals) else ""
            rows.append({"cluster":int(cluster),"cause_code":code,"count":int(n),"description":desc,"actor":actor})
    return pd.DataFrame(rows)

def data_quality_warnings(df):
    warnings=[]
    for field in ["county","vehicle_type","victim","gender","cause_code"]:
        if field not in df: continue
        s=df[field].fillna("MISSING").astype(str).str.strip()
        unique=s.nunique()
        # crude duplicate-looking values: case/space-normalized collisions are evidence of inconsistent source coding
        norm=s.str.upper().str.replace(r"\\s+"," ",regex=True)
        collisions=int(norm.nunique())
        if unique-collisions>0:
            warnings.append({"field":field,"issue":"Multiple source spellings/format variants likely present","raw_unique":unique,"normalized_unique":collisions})
    missing_cols=[c for c in ["age","hour","minute","cause_code"] if c in df]
    for c in missing_cols:
        n=int(df[c].isna().sum())
        warnings.append({"field":c,"issue":"Missing/invalid values retained for QA-aware modelling","missing_or_invalid":n,"raw_unique":"","normalized_unique":""})
    warnings.append({"field":"cluster","issue":"K-Means cluster structure is strongly influenced by time_period and source-category encoding; rerun after categorical harmonization before treating clusters as substantive latent groups","missing_or_invalid":"","raw_unique":"","normalized_unique":""})
    return pd.DataFrame(warnings)
