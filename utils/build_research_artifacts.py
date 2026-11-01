"""Build the research files consumed by the dashboard from an NTSA workbook."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from utils.research_schema import engineer_ntsa_features
from utils.workbook import _combine_excel_sheets

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_DIR.parent / "2018 to 2023 request.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data"
DEFAULT_EVALUATION = PROJECT_DIR / "v10_detailed_evaluation.json"
DEFAULT_CLUSTERING_RESULTS = DEFAULT_OUTPUT_DIR / "NTSA_2018_2023_clustering_results.json"


def _classification_results(
    dataset: Path,
    evaluation_path: Path,
) -> list[dict]:
    if not evaluation_path.exists():
        return []

    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluated_dataset = evaluation.get("dataset")
    if not evaluated_dataset or Path(evaluated_dataset).resolve() != dataset.resolve():
        return []

    evaluated_models = evaluation.get("classifier_results")
    if evaluated_models:
        return evaluated_models

    report = evaluation.get("classification_report", {})
    weighted = report.get("weighted avg", {})
    accuracy = report.get("accuracy")
    if accuracy is None or not weighted:
        return []

    return [{
        "model": "Random Forest",
        "target": "cause_category",
        "accuracy": accuracy,
        "precision": weighted.get("precision"),
        "recall": weighted.get("recall"),
        "f1_score": weighted.get("f1-score"),
        "cv_f1_weighted": evaluation.get("cv_f1_weighted"),
    }]


def _clustering_results(
    results_path: Path,
) -> tuple[list[dict], list[dict]]:
    if not results_path.is_file():
        return [], []

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    results = []
    sensitivity = []
    for result in payload.get("results", []):
        quality = result.get("quality_metrics") or {}
        parameters = result.get("parameters") or {}
        trial_values = parameters.get("trials", [])
        valid_trials = [
            trial for trial in trial_values
            if trial.get("silhouette_score") is not None
        ]
        silhouette = quality.get("silhouette_score")
        if silhouette is None and valid_trials:
            silhouette = max(
                valid_trials,
                key=lambda trial: trial["silhouette_score"],
            )["silhouette_score"]

        results.append({
            "algorithm": result.get("algorithm", "unknown"),
            "status": result.get("status", "unknown"),
            "n_clusters": result.get("n_clusters"),
            "noise_count": result.get("noise_count", 0),
            "records": result.get("records", payload.get("rows_used", 0)),
            "silhouette": silhouette,
            "calinski_harabasz": quality.get("calinski_harabasz"),
            "davies_bouldin": quality.get("davies_bouldin"),
            "metric_note": quality.get("note"),
            "sampling": parameters.get("sampling", "all rows"),
            "computational_sample_size": parameters.get(
                "computational_sample_size",
                payload.get("rows_used", 0),
            ),
        })
        for trial in valid_trials:
            sensitivity.append({
                "algorithm": result.get("algorithm", "unknown"),
                "k": trial.get("k"),
                "silhouette": trial["silhouette_score"],
                "calinski_harabasz": trial.get("calinski_harabasz"),
                "davies_bouldin": trial.get("davies_bouldin"),
            })

    return results, sensitivity


def build_research_artifacts(
    dataset: Path = DEFAULT_DATASET,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    evaluation_path: Path = DEFAULT_EVALUATION,
    clustering_results_path: Path = DEFAULT_CLUSTERING_RESULTS,
) -> dict:
    """Write the dashboard dataset and manifest, returning the manifest."""
    dataset = dataset.resolve()
    if not dataset.is_file():
        raise FileNotFoundError(f"NTSA workbook not found: {dataset}")

    records, cause_code_mapping = _combine_excel_sheets(dataset.read_bytes())
    records = records.drop_duplicates().reset_index(drop=True)
    engineered = engineer_ntsa_features(
        records,
        cause_code_mapping=cause_code_mapping,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_output = output_dir / "NTSA_2018_2023_feature_engineered.csv"
    engineered.to_csv(dataset_output, index=False)
    clustering, clustering_sensitivity = _clustering_results(
        clustering_results_path
    )

    def count_missing(column: str) -> int:
        return int(engineered[column].isna().sum()) if column in engineered else 0

    manifest = {
        "dataset": {
            "source": dataset.name,
            "records": len(engineered),
            "cause_code_mapping_source": (
                "embedded workbook mapping sheet"
                if cause_code_mapping is not None
                else "project CSV mapping"
            ),
        },
        "data_quality": {
            "date_missing": count_missing("date"),
            "date_recovered": int(engineered["date_recovered"].sum())
            if "date_recovered" in engineered else 0,
            "date_day_conflicts": int(engineered["date_day_conflict"].sum())
            if "date_day_conflict" in engineered else 0,
            "time_missing": int(engineered["time_missing"].sum())
            if "time_missing" in engineered else 0,
            "age_missing_or_invalid": int(engineered["age_missing"].sum())
            if "age_missing" in engineered else count_missing("age"),
            "cause_code_missing": count_missing("cause_code"),
            "mapped_cause_descriptions": int(engineered["cause_description"].notna().sum())
            if "cause_description" in engineered else 0,
        },
        "clustering": clustering,
        "clustering_sensitivity": clustering_sensitivity,
        "classification": _classification_results(dataset, evaluation_path),
        "feature_importance": [],
        "notes": [
            "Clustering metrics are included only when generated from a completed clustering run.",
            "Classification metrics are included only when the saved evaluation identifies this exact source workbook.",
            "Feature-importance results require a separately generated analysis.",
        ],
    }
    manifest_output = output_dir / "research_results_manifest.json"
    manifest_output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the NTSA dataset and research manifest used by the dashboard."
    )
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument(
        "--clustering-results",
        type=Path,
        default=DEFAULT_CLUSTERING_RESULTS,
        help="Optional JSON summary of a completed clustering job.",
    )
    args = parser.parse_args()
    result = build_research_artifacts(
        dataset=args.dataset,
        output_dir=args.output_dir,
        evaluation_path=args.evaluation,
        clustering_results_path=args.clustering_results,
    )
    print(f"Created {args.output_dir / 'NTSA_2018_2023_feature_engineered.csv'}")
    print(f"Created {args.output_dir / 'research_results_manifest.json'}")
    print(
        "Classification metrics included:",
        bool(result["classification"]),
    )
