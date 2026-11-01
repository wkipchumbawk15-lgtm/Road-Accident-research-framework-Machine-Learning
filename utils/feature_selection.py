"""Research feature-selection policy for the NTSA accident framework.

The selection policy keeps raw/source fields for traceability but separates them
from modelling predictors. Supervised selection is intentionally target-dependent;
therefore no final RF/SVM feature ranking is performed until the classification
target is approved.
"""

TRACEABILITY_FIELDS = [
    "date", "day", "time_24_hours", "analysis_date", "source_sheet",
    "date_explicit", "date_recovered", "date_day_conflict"
]

CAUSE_DERIVED_FIELDS = ["cause_description", "cause_actor", "mapping_source"]

RAW_HIGH_CARDINALITY_FIELDS = ["road", "place"]

REDUNDANT_DERIVED_FIELDS = [
    # Retain day_of_week as the canonical weekly temporal representation.
    "is_weekend",
    # Retain numeric age; age_group is retained only as a categorical alternative
    # for clustering and interpretation, not together with age by default.
    "age_group",
]

QUALITY_FLAGS = ["time_missing", "age_missing"]

BASE_NUMERIC_CLUSTER_FEATURES = [
    "year", "month", "day_of_week", "hour", "minute", "age", "no"
]

BASE_CATEGORICAL_CLUSTER_FEATURES = [
    "county", "vehicle_type", "gender", "victim", "time_period"
]

# Optional alternatives for sensitivity experiments.
OPTIONAL_CLUSTER_FEATURES = ["road", "place", "age_group", "is_weekend"]


def clustering_feature_set(include_high_cardinality=False, include_age_group=False):
    features = BASE_NUMERIC_CLUSTER_FEATURES.copy()
    features += BASE_CATEGORICAL_CLUSTER_FEATURES.copy()
    if include_age_group:
        features.append("age_group")
    if include_high_cardinality:
        features += RAW_HIGH_CARDINALITY_FIELDS
    return features


def classification_candidate_features():
    """Return target-independent candidates.

    Final supervised selection must happen after a target is approved and only
    on the training partition, using RFE/model importance as described in the
    proposal.
    """
    return [
        "year", "month", "day_of_week", "hour", "minute", "age", "no",
        "county", "vehicle_type", "gender", "victim", "time_period"
    ]


def excluded_feature_reasons():
    return {
        **{x: "Traceability/source field; not a modelling predictor." for x in TRACEABILITY_FIELDS},
        **{x: "Cause-derived field; exclude when cause-related outcome is the target to prevent target leakage." for x in CAUSE_DERIVED_FIELDS},
        "road": "High cardinality; retain for spatial analysis and sensitivity experiments rather than the initial model set.",
        "place": "Very high cardinality; retain for spatial analysis and sensitivity experiments rather than the initial model set.",
        "is_weekend": "Derived from day_of_week; redundant in the initial set.",
        "age_group": "Derived from age; redundant when numeric age is retained. Can be tested as a categorical sensitivity feature.",
        "time_missing": "Data-quality flag; retained for QA analysis, not a substantive accident factor in the initial model set.",
        "age_missing": "Data-quality flag; retained for QA analysis, not a substantive accident factor in the initial model set.",
        "cause_code": "Outcome/cause field; target-dependent. Exclude from predictors when predicting cause and retain for unsupervised descriptive analysis."
    }
