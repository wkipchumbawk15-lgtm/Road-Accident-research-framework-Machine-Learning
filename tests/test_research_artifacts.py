import json

import pandas as pd

from utils.build_research_artifacts import build_research_artifacts


def test_build_research_artifacts_creates_dashboard_files_from_embedded_mapping(tmp_path):
    workbook = tmp_path / "ntsa.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame(
            [
                {
                    "DATE": "2023-01-02",
                    "DAY": "MONDAY",
                    "TIME 24 HOURS": 830,
                    "COUNTY": "NAIROBI",
                    "ROAD": "RING ROAD",
                    "PLACE": "CBD",
                    "TYPE OF VEHICLE": "BUS",
                    "GENDER": "M",
                    "AGE": 27,
                    "CAUSE CODE": 1,
                    "VICTIM": "PED",
                    "NO.": 1,
                }
            ]
        ).to_excel(writer, sheet_name="2023", index=False)
        pd.DataFrame(
            [{
                "Category": "Driver Errors",
                "Code": 1,
                "Description": "Turning right",
            }]
        ).to_excel(writer, sheet_name="NTSA Cause Codes", index=False)

    output_dir = tmp_path / "data"
    clustering_results = tmp_path / "clustering.json"
    clustering_results.write_text(
        json.dumps({
            "rows_used": 1,
            "results": [
                {
                    "algorithm": "kmeans",
                    "status": "done",
                    "n_clusters": 2,
                    "noise_count": 0,
                    "quality_metrics": {
                        "silhouette_score": 0.25,
                        "calinski_harabasz": 3.0,
                        "davies_bouldin": 1.2,
                    },
                    "parameters": {
                        "trials": [
                            {
                                "k": 2,
                                "silhouette_score": 0.25,
                                "calinski_harabasz": 3.0,
                                "davies_bouldin": 1.2,
                            }
                        ]
                    },
                },
                {
                    "algorithm": "dbscan",
                    "status": "done",
                    "n_clusters": 0,
                    "noise_count": 1,
                    "quality_metrics": {
                        "note": "All records were classified as noise."
                    },
                    "parameters": {"eps": 0.5, "min_samples": 5},
                },
            ],
        }),
        encoding="utf-8",
    )
    manifest = build_research_artifacts(
        dataset=workbook,
        output_dir=output_dir,
        evaluation_path=tmp_path / "missing_evaluation.json",
        clustering_results_path=clustering_results,
    )

    engineered = pd.read_csv(output_dir / "NTSA_2018_2023_feature_engineered.csv")
    saved_manifest = json.loads(
        (output_dir / "research_results_manifest.json").read_text(encoding="utf-8")
    )
    assert len(engineered) == 1
    assert engineered.loc[0, "cause_category"] == "Driver Errors"
    assert manifest["dataset"]["cause_code_mapping_source"] == "embedded workbook mapping sheet"
    assert saved_manifest["dataset"]["records"] == 1
    assert saved_manifest["data_quality"]["mapped_cause_descriptions"] == 1
    assert saved_manifest["classification"] == []
    assert saved_manifest["clustering"][0]["silhouette"] == 0.25
    assert saved_manifest["clustering"][1]["silhouette"] is None
    assert saved_manifest["clustering"][1]["noise_count"] == 1
    assert saved_manifest["clustering_sensitivity"][0]["k"] == 2

    evaluation_path = tmp_path / "evaluation.json"
    evaluated_models = [
        {
            "model": "Random Forest",
            "target": "cause_category",
            "accuracy": 0.7,
            "precision": 0.7,
            "recall": 0.7,
            "f1_score": 0.7,
            "cv_f1_weighted": 0.68,
        },
        {
            "model": "SVM",
            "target": "cause_category",
            "accuracy": 0.6,
            "precision": 0.6,
            "recall": 0.6,
            "f1_score": 0.6,
            "cv_f1_weighted": 0.58,
        },
    ]
    evaluation_path.write_text(
        json.dumps({
            "dataset": str(workbook.resolve()),
            "classifier_results": evaluated_models,
        }),
        encoding="utf-8",
    )
    manifest_with_classifiers = build_research_artifacts(
        dataset=workbook,
        output_dir=output_dir,
        evaluation_path=evaluation_path,
        clustering_results_path=clustering_results,
    )
    assert manifest_with_classifiers["classification"] == evaluated_models
