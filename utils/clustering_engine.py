"""Research clustering engine for the NTSA accident framework."""
from __future__ import annotations

from typing import Dict, Iterable
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, estimate_bandwidth
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from utils.feature_engineering import build_cluster_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES

SUPPORTED_ALGORITHMS = ["kmeans", "dbscan", "agglomerative", "mean_shift"]
DEFAULT_FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


def _metrics(X, labels):
    labels = np.asarray(labels)
    mask = labels != -1
    Xv, lv = X[mask], labels[mask]
    unique = np.unique(lv)
    if len(unique) < 2 or len(lv) <= len(unique):
        return {"note": "At least two non-noise clusters with sufficient observations are required."}
    return {
        "silhouette_score": round(float(silhouette_score(Xv, lv)), 6),
        "calinski_harabasz": round(float(calinski_harabasz_score(Xv, lv)), 6),
        "davies_bouldin": round(float(davies_bouldin_score(Xv, lv)), 6),
    }


def _result(name, labels, metrics, params, row_indices):
    labels = np.asarray(labels)
    clusters = [int(x) for x in sorted(set(labels)) if x != -1]
    sizes = {str(c): int((labels == c).sum()) for c in clusters}
    assignments = [{"row_index": int(i), "cluster": int(label)} for i, label in zip(row_indices, labels)]
    return {
        "algorithm": name,
        "status": "done",
        "n_clusters": len(clusters),
        "noise_count": int((labels == -1).sum()),
        "cluster_sizes": sizes,
        "quality_metrics": metrics,
        "parameters": params,
        "assignments": assignments,
    }


def _matrix(df: pd.DataFrame):
    preprocessor, numeric, categorical = build_cluster_preprocessor(df, include_high_cardinality=False)
    # Ensure categorical missing values are filled before fitting the preprocessor
    if categorical:
        df_for_fit = df.copy()
        for col in categorical:
            if col in df_for_fit.columns:
                df_for_fit[col] = df_for_fit[col].astype("string").fillna("MISSING")
    else:
        df_for_fit = df

    X = preprocessor.fit_transform(df_for_fit)
    # One-hot encoded mixed data can become high dimensional. Compress only when
    # necessary so distance-based clustering remains computationally practical.
    transformed_features = int(X.shape[1])
    reduction = None
    if transformed_features > 30:
        n_components = min(30, transformed_features - 1, max(2, len(df) - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X = svd.fit_transform(X)
        reduction = {"method": "TruncatedSVD", "n_components": int(n_components),
                     "explained_variance_ratio_sum": round(float(svd.explained_variance_ratio_.sum()), 6)}
    else:
        X = X.toarray() if hasattr(X, "toarray") else X
    X = StandardScaler().fit_transform(X)
    return X, numeric, categorical, transformed_features, reduction


def run_clustering(df: pd.DataFrame, algorithms: Iterable[str] | None = None,
                    features: Iterable[str] | None = None, dbscan_eps=0.5,
                    dbscan_min_samples=5, computational_sample_size=5000,
                    mean_shift_bandwidth=None) -> Dict:
    algorithms = [a.lower() for a in (algorithms or SUPPORTED_ALGORITHMS)]
    algorithms = [a for a in algorithms if a in SUPPORTED_ALGORITHMS]
    if not algorithms:
        raise ValueError(f"No supported clustering algorithms selected. Supported: {SUPPORTED_ALGORITHMS}")

    # `features` is retained for API compatibility. The research clustering matrix
    # is deliberately mixed-type and uses the canonical numeric/categorical feature set.
    work = df.copy().reset_index(drop=False).rename(columns={"index": "_source_row"})
    X, numeric, categorical, original_dim, reduction = _matrix(work)
    valid_rows = work.index.to_numpy()
    if len(X) < 3:
        raise ValueError("Not enough observations for clustering.")

    results = []
    best_k = None
    k_trials = []
    if "kmeans" in algorithms:
        max_k = min(8, len(X) - 1)
        for k in range(2, max_k + 1):
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X)
            m = _metrics(X, labels)
            k_trials.append({"k": k, **m})
        valid = [x for x in k_trials if "silhouette_score" in x]
        best_k = max(valid, key=lambda x: x["silhouette_score"])["k"] if valid else 2
        model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        results.append(_result("kmeans", labels, _metrics(X, labels),
                               {"n_clusters": best_k, "selection": "maximum silhouette", "trials": k_trials},
                               work.loc[valid_rows, "_source_row"].tolist()))

    if "dbscan" in algorithms:
        model = DBSCAN(eps=float(dbscan_eps), min_samples=int(dbscan_min_samples))
        labels = model.fit_predict(X)
        results.append(_result("dbscan", labels, _metrics(X, labels),
                               {"eps": float(dbscan_eps), "min_samples": int(dbscan_min_samples)},
                               work.loc[valid_rows, "_source_row"].tolist()))

    # Agglomerative and Mean Shift have substantially higher computational cost.
    # For reproducibility on the full NTSA population, use a deterministic county-
    # proportional sample for these algorithms and retain the sampling metadata.
    expensive_algorithms = {a for a in algorithms if a in {"agglomerative", "mean_shift"}}
    expensive_idx = None
    X_expensive = X
    if expensive_algorithms and len(work) > computational_sample_size:
        rng = np.random.default_rng(42)
        # Ensure `county` column exists and contains no null categories before grouping.
        if "county" not in work.columns:
            work["county"] = pd.Series(["UNKNOWN_COUNTY"] * len(work), index=work.index)
        work["county"] = work["county"].astype("string").fillna("UNKNOWN_COUNTY")
        work["county"] = work["county"].str.strip().replace("", "UNKNOWN_COUNTY")
        groups = work.groupby("county", dropna=False, sort=True).groups
        chosen = []
        for _, idxs in groups.items():
            idxs = np.asarray(list(idxs))
            n = max(1, int(round(len(idxs) / len(work) * computational_sample_size)))
            n = min(n, len(idxs))
            chosen.extend(rng.choice(idxs, size=n, replace=False).tolist())
        if len(chosen) > computational_sample_size:
            chosen = rng.choice(np.asarray(chosen), size=computational_sample_size, replace=False).tolist()
        expensive_idx = np.sort(np.asarray(chosen, dtype=int))
        X_expensive = X[expensive_idx]

    if "agglomerative" in algorithms:
        k_candidates = [best_k or 2] if best_k else [2, 3, 4, 5, 6]
        agg_trials = []
        for k in k_candidates:
            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(X_expensive)
            agg_trials.append({"k": k, **_metrics(X_expensive, labels)})
        valid = [x for x in agg_trials if "silhouette_score" in x]
        k = max(valid, key=lambda x: x["silhouette_score"])["k"] if valid else 2
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(X_expensive)
        source = work.loc[valid_rows[expensive_idx], "_source_row"].tolist() if expensive_idx is not None else work.loc[valid_rows, "_source_row"].tolist()
        results.append(_result("agglomerative", labels, _metrics(X_expensive, labels),
                               {"n_clusters": k, "selection": "maximum silhouette", "trials": agg_trials,
                                "computational_sample_size": int(len(X_expensive)), "sampling": "county-proportional deterministic sample"}, source))

    if "mean_shift" in algorithms:
        Xms = X_expensive
        bandwidth = mean_shift_bandwidth
        bandwidth_selection = "user-specified" if bandwidth is not None else "estimated from 500-row deterministic sample, quantile=0.2"
        if bandwidth is None:
            rng = np.random.default_rng(42)
            probe = Xms[rng.choice(len(Xms), size=min(500, len(Xms)), replace=False)]
            bandwidth = float(estimate_bandwidth(probe, quantile=0.2, n_samples=len(probe), random_state=42))
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            bandwidth = 1.0
        # Evaluate a small, reproducible bandwidth neighborhood rather than relying
        # on MeanShift's default, which can collapse this dataset into one cluster.
        bw_trials = [float(bandwidth * m) for m in (0.75, 1.0, 1.25)]
        ms_trials = []
        for bw in bw_trials:
            model = MeanShift(bandwidth=bw, bin_seeding=True, n_jobs=-1)
            lab = model.fit_predict(Xms)
            ms_trials.append({"bandwidth": bw, "n_clusters": int(len(np.unique(lab))), **_metrics(Xms, lab)})
        valid = [x for x in ms_trials if "silhouette_score" in x]
        chosen_bw = max(valid, key=lambda x: x["silhouette_score"])["bandwidth"] if valid else bw_trials[1]
        model = MeanShift(bandwidth=chosen_bw, bin_seeding=True, n_jobs=-1)
        labels = model.fit_predict(Xms)
        source = work.loc[valid_rows[expensive_idx], "_source_row"].tolist() if expensive_idx is not None else work.loc[valid_rows, "_source_row"].tolist()
        results.append(_result("mean_shift", labels, _metrics(Xms, labels),
                               {"bandwidth": chosen_bw, "bandwidth_selection": bandwidth_selection, "trials": ms_trials,
                                "computational_sample_size": int(len(Xms)), "sampling": "county-proportional deterministic sample"}, source))

    return {
        "feature_set": {"numeric": numeric, "categorical": categorical},
        "transformed_dimensions": original_dim,
        "dimensionality_reduction": reduction,
        "rows_used": int(len(X)),
        "results": results,
    }
