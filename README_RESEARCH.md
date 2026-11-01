# Research-Aligned Accident System

This version is the implementation baseline for the JKUAT NTSA road-accident research framework.

## Start

1. Build the services with Docker Compose.
2. Open the Streamlit dashboard.
3. Upload the NTSA CSV/XLSX/XLS dataset.
4. Wait for preprocessing to finish.
5. The proposal-aligned classification target is `cause_category`, which groups accidents by mapped cause.
6. Run the research workflow.
7. Inspect clustering metrics, classification metrics, and experiment metadata.

## Important

The NTSA dataset does not contain `accident_severity`. The framework does not manufacture severity labels; its cause-classification workflow uses the mapped `cause_category` outcome described below.

Do not treat cause-derived fields as predictors when the selected target is a cause-derived outcome; doing so can introduce target leakage.

The existing synthetic `data/road_accidents.csv` is retained only as a development reference and is not used by the research training job.

When an uploaded Excel workbook contains a sheet with `Category`, `Code`, and `Description` columns, preprocessing uses that sheet as the cause-code mapping, excludes it from accident records, and assigns its category directly to `cause_category`. This workbook mapping takes precedence over the bundled CSV mappings. Unmatched or missing cause codes remain uncategorized and are excluded from classification. For CSV files or Excel workbooks without an embedded mapping sheet, preprocessing retains the bundled `utils/cause_code_mapping.csv` and provisional `utils/cause_category_mapping.csv` fallback. Re-upload and preprocess after changing a mapping.

The proposal's cause-categorization objective is implemented with `cause_category` as the classification outcome and Random Forest/SVM as the classifiers. `DATE` is retained only for deriving temporal predictors and is rejected as a classification target. The available NTSA workbook does not include geospatial coordinates or environmental measurements; those proposal factors are not fabricated and require approved supplementary data.

The reproducible classifier comparison evaluates Random Forest and Linear SVM
on the same stratified 80/20 holdout and uses weighted-F1 stratified
cross-validation for hyperparameter selection. It is a research comparison and
does not replace or activate a model in the serving registry.
The dashboard training option `svm` uses this Linear SVM implementation to keep
cross-validation practical on the full dataset. Its predictions include class
labels but not calibrated probabilities.

The RQ worker allows up to one hour for long training jobs and has a 65-minute
Docker shutdown grace period so a normal container restart can finish an active
job. Training locks are owned by their RQ job ID and are released only by that
owner; the status endpoint also clears a failed job's matching stale lock.

Preprocessing normalizes the source workbook's uppercase NTSA headers before feature engineering and cause mapping. After deploying this change, upload and preprocess the workbook again; previously processed CSV files do not automatically gain the category target or corrected temporal features.

## Research alignment and data readiness

| Proposal component | Current implementation | Research action |
|---|---|---|
| Identify and categorize accident causes | Classifies mapped `cause_category` using Random Forest and SVM | Describe this as cause-category classification among recorded cases, not accident-severity prediction or causal proof. |
| K-Means, DBSCAN, Agglomerative, Mean Shift | All four clustering methods are implemented | Report sampling and algorithm-specific limitations; Agglomerative and Mean Shift may use county-proportional samples. |
| Classifiers | Runtime supports Random Forest and SVM. The proposal's abstract/methodology also names Multinomial Naive Bayes and RNN, while its specific objective names RF/SVM. | Resolve this proposal inconsistency with the supervisor. Until then, report only RF/SVM as implemented; `utils/train_rnn.py` is not part of the training workflow. |
| Feature selection/reduction | Classification uses `SelectPercentile(f_classif)`; clustering uses TruncatedSVD when the encoded matrix is wide. | The proposal also mentions RFE/feature importance and PCA. Record these as implementation choices/deviations or revise and approve the methods; do not claim PCA/RFE was run. |
| Normalization | Numeric classification/clustering features use `StandardScaler`. | The proposal specifies Min-Max normalization. Choose and justify one method, then make the code, methods chapter, and reported experiments consistent. |
| Geospatial/environmental factors and integrated sources | The workbook has county, road and place text, but no coordinates, weather, road condition, vehicle registry or traffic-surveillance measurements. | Obtain approved supplementary data and document linkage/quality, or state these analyses were not possible with the available data. Do not fabricate fields. |

The supplied workbook has 22,114 accident rows in six yearly sheets (2018–2023), plus an `NTSA Cause Codes` mapping sheet with 90 code descriptions in nine categories. In a direct code-coverage check, 17,850 accident rows (80.7%) matched that mapping; the remainder include missing, malformed, and unmapped codes. Actual counts are recorded in preprocessing metadata for each upload. Review coverage and class balance before evaluating models; do not merge or remove categories without a documented, approved rule.

The source has 2,367 missing DATE cells; group-date recovery plus the year sheet labels yield a year for all rows, but month is unavailable for 42 rows. Parsed hour/minute are available for 21,161 rows; the rest have missing or invalid time values. AGE is frequently recorded as `A` rather than a numeric value, so age-based groups/indicators must be reported as having substantial missingness. ROAD and PLACE are high-cardinality free text (10,434 and 15,829 distinct raw values, respectively) and need documented harmonization before interpreting location patterns.

The workbook does not contain a crash/event identifier, an explicit severity/injury/fatality outcome, or non-crash exposure (such as traffic volume). Without these, it cannot support event-grouped validation, severity prediction, or estimating crash occurrence/risk per road user or trip. Current classification predicts a mapped cause category conditional on a row already being in the accident workbook.

The prediction interface uses numeric widgets for numeric predictors and model-learned categorical options for categorical predictors; road and place are free-text fields. Categorical values must be strings, not numeric codes. Prediction probabilities are displayed against the corresponding learned class labels.

The small data frames in `tests/test_train_model_schema.py` are unit-test fixtures, not research test sets. The two-row `severity` fixture only checks loader behavior; it is not evidence of a valid severity label or model performance. For final results, define the unit of analysis, mapping version, inclusion criteria, and a held-out evaluation design before fitting; use a year-based holdout (for example, train through 2022 and test on 2023) if the research claim is performance on later years, and preserve enough examples of each class to evaluate it.


## v12 Integrated Decision-Support Dashboard
The Streamlit dashboard now includes a Research Results tab driven by
`/data/research_results_manifest.json` and the engineered NTSA dataset.
The dashboard container mounts `./data:/data`, while the existing upload,
training, clustering and prediction workflows remain available.
The clustering dataset selector lists uploaded CSV/Excel data files only,
remembers the selected file across reruns, and defaults to the most recently
uploaded file in the current dashboard session.

To create the research files for the supplied workbook, run
`python -m utils.build_research_artifacts` from the `accident-system` directory.
This writes `data/NTSA_2018_2023_feature_engineered.csv` and
`data/research_results_manifest.json`, using an embedded workbook cause-code
mapping when available. Saved classification metrics are included only when
the existing detailed evaluation refers to that exact workbook. Clustering and
feature-importance evidence are not fabricated by this builder and remain
blank until those experiments are generated. To include a completed clustering
job, provide its summary JSON with `--clustering-results`; the default is
`data/NTSA_2018_2023_clustering_results.json`. The generated result summary
includes quality metrics only where the clustering run produced valid scores.
