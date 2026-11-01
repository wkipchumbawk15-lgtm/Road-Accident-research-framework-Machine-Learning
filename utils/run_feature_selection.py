"""Generate the reproducible NTSA feature-selection report."""
from pathlib import Path
import pandas as pd
import numpy as np
from feature_selection import (
    TRACEABILITY_FIELDS, CAUSE_DERIVED_FIELDS, RAW_HIGH_CARDINALITY_FIELDS,
    REDUNDANT_DERIVED_FIELDS, QUALITY_FLAGS, BASE_NUMERIC_CLUSTER_FEATURES,
    BASE_CATEGORICAL_CLUSTER_FEATURES, classification_candidate_features,
    clustering_feature_set, excluded_feature_reasons,
)

INPUT = Path('/data/NTSA_2018_2023_feature_engineered.csv')
OUTPUT = Path('/data/NTSA_2018_2023_feature_selection_report.xlsx')

# Also works locally when executed from the repository root.
if not INPUT.exists():
    INPUT = Path('/mnt/data/NTSA_2018_2023_feature_engineered.csv')
if not OUTPUT.parent.exists():
    OUTPUT = Path('/mnt/data/NTSA_2018_2023_feature_selection_report.xlsx')

df = pd.read_csv(INPUT)

all_features = list(df.columns)
reasons = excluded_feature_reasons()
rows = []
for f in all_features:
    nonmissing = df[f].notna().sum()
    rows.append({
        'feature': f,
        'dtype': str(df[f].dtype),
        'unique_values': int(df[f].nunique(dropna=True)),
        'missing_count': int(df[f].isna().sum()),
        'missing_pct': round(float(df[f].isna().mean()*100), 3),
        'role': ('candidate_cluster' if f in clustering_feature_set() else
                 'candidate_classification' if f in classification_candidate_features() else
                 'excluded_or_target_dependent'),
        'decision': ('retain_initial' if f in clustering_feature_set() else
                     'retain_for_supervised_selection' if f in classification_candidate_features() else
                     'exclude_from_initial_predictors'),
        'reason': reasons.get(f, 'Candidate feature; final supervised selection is target-dependent.')
    })

catalog = pd.DataFrame(rows)

numeric = df.select_dtypes(include=[np.number]).columns.tolist()
corr = df[numeric].corr(method='spearman').round(4)

categorical_cols = [c for c in all_features if df[c].dtype == 'object']
card = pd.DataFrame([
    {'feature': c, 'unique_values': int(df[c].nunique(dropna=True)),
     'missing_pct': round(float(df[c].isna().mean()*100), 3),
     'recommendation': ('retain' if c in BASE_CATEGORICAL_CLUSTER_FEATURES else
                        'sensitivity/high-cardinality' if c in RAW_HIGH_CARDINALITY_FIELDS else
                        'target-dependent/derived' if c in CAUSE_DERIVED_FIELDS + ['cause_code','age_group'] else
                        'traceability')}
    for c in categorical_cols
])

cluster = pd.DataFrame({'feature': clustering_feature_set(), 'role': 'initial clustering'})
classification = pd.DataFrame({'feature': classification_candidate_features(),
                                'role': 'candidate; target-dependent selection pending'})
excluded = pd.DataFrame([
    {'feature': f, 'reason': reasons.get(f, 'Excluded from initial predictors.')} for f in all_features
    if f not in clustering_feature_set() and f not in classification_candidate_features()
])

summary = pd.DataFrame([
    ['Rows', len(df)],
    ['Engineered columns', len(all_features)],
    ['Initial clustering features', len(cluster)],
    ['Classification candidates before target-dependent selection', len(classification)],
    ['High-cardinality spatial fields', len(RAW_HIGH_CARDINALITY_FIELDS)],
    ['Cause-derived fields excluded from predictors', len(CAUSE_DERIVED_FIELDS)],
    ['Final supervised target', 'NOT APPROVED — selection remains provisional'],
], columns=['item','value'])

with pd.ExcelWriter(OUTPUT, engine='openpyxl') as writer:
    summary.to_excel(writer, 'Summary', index=False)
    catalog.to_excel(writer, 'Feature Catalog', index=False)
    excluded.to_excel(writer, 'Excluded Features', index=False)
    corr.to_excel(writer, 'Numeric Correlations')
    card.to_excel(writer, 'Categorical Cardinality', index=False)
    cluster.to_excel(writer, 'Clustering Features', index=False)
    classification.to_excel(writer, 'Classification Candidates', index=False)
    pd.DataFrame({
        'feature': BASE_NUMERIC_CLUSTER_FEATURES + BASE_CATEGORICAL_CLUSTER_FEATURES,
        'status': 'selected_initial',
        'note': 'Unsupervised initial set; evaluate with clustering metrics.'
    }).to_excel(writer, 'Final Provisional Set', index=False)

print(OUTPUT)
