from pathlib import Path
import joblib
import json
import os
from datetime import datetime
import re

# Use absolute path so containers can share /models when mounted
MODEL_DIR = Path("/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
ACTIVE_MODEL_FILE = MODEL_DIR / "active_model.txt"
VERSIONS_FILE = MODEL_DIR / "versions.json"

# Registry of available models (in-memory cache)
MODEL_REGISTRY = {}

# Active / selected model (cached)
ACTIVE_MODEL = None


def _load_versions():
    """Load version metadata from disk."""
    if VERSIONS_FILE.exists():
        content = VERSIONS_FILE.read_text(encoding="utf-8")
        if not content.strip():
            return {}
        versions = json.loads(content)
        if not isinstance(versions, dict):
            raise ValueError("Model versions registry must contain a JSON object.")
        return versions
    return {}


def _save_versions(versions):
    """Save version metadata to disk."""
    VERSIONS_FILE.write_text(json.dumps(versions, indent=2))


def create_version(classifiers, clusterers, classification_results, clustering_results=None, dataset_object=None, target_column=None, preprocessing=None, feature_selection=None):
    """Create a new model version with metadata.
    
    Args:
        classifiers: list of classifier names used
        clusterers: list of clusterer names used
        classification_results: dict of classification metrics
        clustering_results: dict of clustering metrics (optional)
    
    Returns:
        version_name: e.g., "v1", "v2", etc.
    """
    versions = _load_versions()
    version_num = len(versions) + 1
    version_name = f"v{version_num}"
    
    # Find best classifier by f1_score
    best_classifier = None
    best_f1 = -1
    for clf_name, metrics in classification_results.items():
        f1 = metrics.get("test_metrics", {}).get("f1_score", 0)
        if f1 > best_f1:
            best_f1 = f1
            best_classifier = clf_name
    
    versions[version_name] = {
        "timestamp": datetime.now().isoformat(),
        "classifiers": classifiers,
        "clusterers": clusterers,
        "best_classifier": best_classifier,
        "classification_results": classification_results,
        "clustering_results": clustering_results or {},
        "dataset_object": dataset_object,
        "target_column": target_column,
        "preprocessing": preprocessing or {},
        "feature_selection": feature_selection or {}
    }
    
    _save_versions(versions)
    return version_name


def get_versions():
    """Get all versions with metadata."""
    return _load_versions()


def get_version_by_name(version_name):
    """Get metadata for a specific version."""
    versions = _load_versions()
    return versions.get(version_name)


def _discover_models_on_disk():
    """Populate MODEL_REGISTRY from joblib files on disk, excluding non-model artifacts."""
    # List of known non-model artifact names to exclude
    ARTIFACTS = {"scaler", "feature_order"}
    
    for p in MODEL_DIR.glob("*.joblib"):
        name = p.stem
        # Skip preprocessing artifacts (scaler, feature_order, etc.)
        if name in ARTIFACTS or name.startswith("_"):
            continue
        if name not in MODEL_REGISTRY:
            MODEL_REGISTRY[name] = {
                "type": "Unknown",
                "task": "classification"
            }


def register_model(name: str, model_or_type, task: str = None):
    """Register a model.

    Backwards-compatible behavior:
    - register_model(name, model_obj) -> saves model as joblib and infers type/task
    - register_model(name, model_type_str, task_str) -> stores metadata only
    """
    if task is None and not isinstance(model_or_type, str):
        # Assume a model object was passed: save to disk and infer metadata
        model = model_or_type
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        MODEL_REGISTRY[name] = {
            "type": model.__class__.__name__,
            "task": "classification"
        }
    else:
        # Old style: explicit type and task strings
        MODEL_REGISTRY[name] = {
            "type": str(model_or_type),
            "task": task
        }


def get_models():
    # Ensure we include files saved directly to disk
    _discover_models_on_disk()
    return list(MODEL_REGISTRY.keys())


def delete_model_artifact(model_name: str) -> bool:
    """Delete a trained classifier artifact and clear it if currently active."""
    global ACTIVE_MODEL

    if not re.fullmatch(r"[A-Za-z0-9_-]+", model_name):
        raise ValueError("Invalid model name")

    model_path = MODEL_DIR / f"{model_name}.joblib"
    if not model_path.is_file():
        MODEL_REGISTRY.pop(model_name, None)
        return False

    model_path.unlink()
    MODEL_REGISTRY.pop(model_name, None)

    active_file_matches = (
        ACTIVE_MODEL_FILE.exists()
        and ACTIVE_MODEL_FILE.read_text().strip() == model_name
    )
    if ACTIVE_MODEL == model_name or active_file_matches:
        ACTIVE_MODEL = None
        ACTIVE_MODEL_FILE.unlink(missing_ok=True)

    return True


def set_active_model(model_name: str):
    global ACTIVE_MODEL

    # Accept model if it exists on disk even if not in registry
    if model_name not in MODEL_REGISTRY:
        if not (MODEL_DIR / f"{model_name}.joblib").exists():
            raise ValueError(f"Model '{model_name}' is not registered or available on disk")
        MODEL_REGISTRY[model_name] = {
            "type": "Unknown",
            "task": "classification"
        }

    # Persist active model selection to disk so other services can read it
    ACTIVE_MODEL_FILE.write_text(model_name)
    ACTIVE_MODEL = model_name


def get_active_model():
    global ACTIVE_MODEL
    # Use cached value if set
    if ACTIVE_MODEL:
        return ACTIVE_MODEL

    # Fall back to persisted file
    if ACTIVE_MODEL_FILE.exists():
        name = ACTIVE_MODEL_FILE.read_text().strip()
        if name:
            ACTIVE_MODEL = name
            return name

    raise ValueError("No active model has been set")
