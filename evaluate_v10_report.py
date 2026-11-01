
import argparse
from pathlib import Path
import json
from tempfile import TemporaryDirectory

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from utils.preprocess_job import _combine_excel_sheets
from utils.research_schema import engineer_ntsa_features
from utils.train_model import load_dataset, _pipeline

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_DIR.parent / "2018 to 2023 request.xlsx"


def evaluate(dataset: Path) -> dict:
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset.resolve()}")

    mapping_source = "existing processed dataset"
    temp_dir = None
    if dataset.suffix.lower() in {".xlsx", ".xls"}:
        records, cause_code_mapping = _combine_excel_sheets(dataset.read_bytes())
        records = records.drop_duplicates().reset_index(drop=True)
        records = engineer_ntsa_features(
            records,
            cause_code_mapping=cause_code_mapping,
        )
        mapping_source = (
            "embedded workbook mapping sheet"
            if cause_code_mapping is not None
            else "project CSV mapping"
        )
        temp_dir = TemporaryDirectory()
        dataset_for_loader = Path(temp_dir.name) / "cleaned_dataset.csv"
        records.to_csv(dataset_for_loader, index=False)
    else:
        dataset_for_loader = dataset

    try:
        X, y, predictors, rows_used = load_dataset(
            str(dataset_for_loader),
            "cause_category",
        )

        # Match the training workflow's treatment of rare classes.
        counts = y.value_counts()
        rare_classes = counts[counts < 2].index

        if len(rare_classes):
            keep = ~y.isin(rare_classes)
            X = X.loc[keep]
            y = y.loc[keep]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )

        # Match the training workflow's cross-validation procedure.
        n_splits = min(5, int(y_train.value_counts().min()))

        if n_splits < 2:
            raise ValueError("Insufficient examples per class for cross-validation.")

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42,
        )

        model_specs = {
            "Random Forest": (
                RandomForestClassifier(
                    random_state=42,
                    n_jobs=1,
                    class_weight="balanced",
                ),
                {
                    "classifier__n_estimators": [100],
                    "classifier__max_depth": [None, 15],
                },
            ),
            "Linear SVM": (
                LinearSVC(
                    random_state=42,
                    class_weight="balanced",
                    dual="auto",
                    max_iter=20000,
                ),
                {"classifier__C": [0.5, 1.0]},
            ),
        }
        classifier_results = []
        detailed_results = {}
        for model_name, (classifier, parameter_grid) in model_specs.items():
            search = GridSearchCV(
                _pipeline(classifier),
                parameter_grid,
                scoring="f1_weighted",
                cv=cv,
                n_jobs=1,
                refit=True,
            )
            print(f"Training temporary {model_name} evaluation model...")
            search.fit(X_train, y_train)

            predictions = search.best_estimator_.predict(X_test)
            labels = search.best_estimator_.named_steps["classifier"].classes_
            matrix = confusion_matrix(y_test, predictions, labels=labels)
            report = classification_report(
                y_test,
                predictions,
                zero_division=0,
                output_dict=True,
            )
            weighted = report["weighted avg"]
            classifier_results.append({
                "model": model_name,
                "target": "cause_category",
                "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
                "precision": round(float(weighted["precision"]), 4),
                "recall": round(float(weighted["recall"]), 4),
                "f1_score": round(float(weighted["f1-score"]), 4),
                "cv_f1_weighted": round(float(search.best_score_), 4),
                "best_params": search.best_params_,
            })
            detailed_results[model_name] = {
                "best_params": search.best_params_,
                "cv_f1_weighted": round(float(search.best_score_), 4),
                "classification_report": report,
                "class_order": list(labels),
                "confusion_matrix": matrix.tolist(),
            }

            print(
                f"{model_name}: accuracy={classifier_results[-1]['accuracy']:.4f}, "
                f"weighted F1={classifier_results[-1]['f1_score']:.4f}, "
                f"CV weighted F1={classifier_results[-1]['cv_f1_weighted']:.4f}"
            )

        random_forest = detailed_results["Random Forest"]
        output = {
            "dataset": str(dataset),
            "cause_code_mapping_source": mapping_source,
            "rows_used": len(y),
            "training_records": len(y_train),
            "test_records": len(y_test),
            "classifier_results": classifier_results,
            "model_details": detailed_results,
            "best_params": random_forest["best_params"],
            "cv_f1_weighted": random_forest["cv_f1_weighted"],
            "classification_report": random_forest["classification_report"],
            "class_order": random_forest["class_order"],
            "confusion_matrix": random_forest["confusion_matrix"],
        }

        print("\nRows used:", len(y))
        print("Training records:", len(y_train))
        print("Test records:", len(y_test))
        print("Categories:", random_forest["class_order"])
        print("Mapping source:", mapping_source)
        for model_name, model_output in detailed_results.items():
            print(f"\n{model_name} CLASSIFICATION REPORT")
            print(json.dumps(model_output["classification_report"], indent=2))
            print("Best parameters:", model_output["best_params"])
            print("Cross-validation weighted F1:", model_output["cv_f1_weighted"])

        output_path = PROJECT_DIR / "v10_detailed_evaluation.json"
        output_path.write_text(
            json.dumps(output, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nSaved results to {output_path}")
        return output
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate cause-category classification using the supplied mapping."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET,
        help="CSV or Excel dataset; defaults to the adjacent NTSA workbook.",
    )
    args = parser.parse_args()
    evaluate(args.dataset)
