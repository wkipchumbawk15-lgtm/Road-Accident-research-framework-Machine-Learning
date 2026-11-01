import json

import pytest

from utils import model_registry


def test_load_versions_treats_empty_registry_as_uninitialized(tmp_path, monkeypatch):
    versions_file = tmp_path / "versions.json"
    versions_file.write_text("", encoding="utf-8")
    monkeypatch.setattr(model_registry, "VERSIONS_FILE", versions_file)

    assert model_registry._load_versions() == {}


def test_load_versions_preserves_valid_registry(tmp_path, monkeypatch):
    versions_file = tmp_path / "versions.json"
    expected = {"v1": {"best_classifier": "random_forest"}}
    versions_file.write_text(json.dumps(expected), encoding="utf-8")
    monkeypatch.setattr(model_registry, "VERSIONS_FILE", versions_file)

    assert model_registry._load_versions() == expected


def test_load_versions_rejects_malformed_nonempty_registry(tmp_path, monkeypatch):
    versions_file = tmp_path / "versions.json"
    versions_file.write_text("{invalid", encoding="utf-8")
    monkeypatch.setattr(model_registry, "VERSIONS_FILE", versions_file)

    with pytest.raises(json.JSONDecodeError):
        model_registry._load_versions()
