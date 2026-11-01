"""Reusable evaluation helpers for research experiments."""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_predictions(y_true, y_pred):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
    }
