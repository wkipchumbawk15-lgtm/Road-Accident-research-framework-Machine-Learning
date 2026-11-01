# NTSA Road Accident Data Mining Framework --- Implementation Baseline

## 1. Purpose

This implementation aligns the existing `accident-system` prototype with
the JKUAT research proposal and the uploaded NTSA 2018--2023 dataset.

The framework retains the existing Docker, FastAPI, Redis/RQ, MinIO and
Streamlit architecture while changing the data and machine-learning flow
so that the research dataset is the source for both clustering and
classification.

## 2. Research pipeline

``` text
NTSA 2018–2023 Workbook
        |
        v
Data Acquisition
        |
        v
Data Quality Assessment
        |
        v
Preprocessing
  - standardize fields
  - remove duplicates
  - preserve missingness indicators
  - parse date/time
  - normalize categorical values
  - clean age
        |
        v
Cause-Code Integration
  - cause code
  - cause description
  - cause actor
  - mapping provenance
        |
        v
Feature Engineering
  - year/month
  - hour/minute
  - day of week
  - weekend
  - time period
  - age group
        |
        v
Feature Selection
  - SelectPercentile / F-test
        |
        +----------------------+
        |                      |
        v                      v
   CLUSTERING             CLASSIFICATION
   K-Means                Random Forest
   DBSCAN                 SVM
   Agglomerative
   Mean Shift
        |                      |
        v                      v
 Cluster Metrics          Accuracy
 Silhouette               Precision
 Calinski-Harabasz        Recall
 Davies-Bouldin           F1
        |                      |
        +----------+-----------+
                   |
                   v
        Latent Pattern Analysis
        - temporal
        - geographic
        - vehicle
        - victim
        - cause
        - cluster profiles
                   |
                   v
        Visualization / Dashboard
                   |
                   v
        Research Interpretation
```

## 3. Dataset alignment

The actual NTSA workbook contains:

-   DATE
-   DAY
-   TIME 24 HOURS
-   COUNTY
-   ROAD
-   PLACE
-   TYPE OF VEHICLE
-   GENDER
-   AGE
-   CAUSE CODE
-   VICTIM
-   NO.

The implementation does **not** create vehicle speed, weather score,
road-condition score, latitude or longitude values when those fields are
absent from the NTSA source.

Environmental or geospatial enrichment should be added only when a
documented external data source is available and the enrichment
procedure is approved for the study.

## 4. Important classification decision

The NTSA dataset does not contain `accident_severity`.

Therefore, the system does not manufacture severity classes. In line
with the proposal's objective to categorize accidents by cause, the
classification workflow uses the provisional `cause_category` derived
from the documented cause mapping. The mapping requires researcher
review before results are interpreted as official cause classes.

`DATE` is used only to derive temporal predictors and cannot be selected
as the classification outcome. Cause-derived fields are excluded from
the predictor set to avoid target leakage. Geospatial coordinates and
environmental factors named in the proposal are not present in the
available NTSA workbook and must not be imputed or fabricated; approved
supplementary data is needed to analyze them.

## 5. Cause-code handling

The uploaded cause-code reference is integrated as a documented mapping.

The processed dataset can contain:

-   `cause_code`
-   `cause_description`
-   `cause_actor`
-   `mapping_source`

Unmapped or uncertain codes remain identifiable rather than being
silently assigned descriptions.

The previously profiled NTSA data contained 22,114 records. The uploaded
profiling workbook reported 18,525 records with usable/non-placeholder
cause codes, 12,047 mapped to the uploaded detailed reference, and 3,664
records still requiring cause-code verification.

## 6. Software changes

### Retained

-   Docker Compose
-   FastAPI API Gateway
-   Redis/RQ
-   MinIO
-   PostgreSQL infrastructure
-   Streamlit dashboard
-   model registry

### Changed

-   Preprocessing supports CSV/XLSX/XLS.
-   Preprocessing is shared through an RQ-importable utility.
-   NTSA-specific feature engineering replaces unsupported synthetic
    features.
-   Clustering uses NTSA-derived numeric features.
-   Cluster results retain row labels and evaluation metrics.
-   Classification trains from the processed research dataset instead of
    the fixed development CSV.
-   Random Forest and SVM are the primary classification models.
-   Feature selection is explicit.
-   Experiment versions store dataset, target, preprocessing,
    feature-selection, classification and clustering metadata.
-   Unsupported Spectral clustering is removed from the research
    workflow.
-   Prediction consumes the same named raw predictors used during
    training.

## 7. Evaluation

Clustering records:

-   Silhouette score
-   Calinski-Harabasz score
-   Davies-Bouldin score
-   number of clusters
-   noise count for DBSCAN
-   cluster sizes
-   algorithm parameters

Classification records:

-   accuracy
-   precision
-   recall
-   F1
-   cross-validation F1
-   best hyperparameters

Scikit-learn documentation describes higher Calinski-Harabasz as
indicating better-defined clusters and lower Davies-Bouldin as better
separation; these measures should be interpreted together rather than
treated as a single universal ranking.

## 8. Research traceability

  -----------------------------------------------------------------------
  Proposal activity                   Implementation
  ----------------------------------- -----------------------------------
  Data collection                     NTSA workbook upload

  Data preprocessing                  Shared preprocessing job

  Data cleaning                       Duplicate and invalid-value
                                      handling

  Data integration                    Cause-code reference integration

  Data transformation                 Date/time, categorical and
                                      age-derived features

  Normalization                       StandardScaler in ML pipelines

  Feature selection                   SelectPercentile

  K-Means                             Implemented

  DBSCAN                              Implemented

  Agglomerative                       Implemented

  Mean Shift                          Implemented

  Random Forest                       Implemented

  SVM                                 Implemented

  Model evaluation                    Classification + clustering metrics

  Visualization                       Streamlit dashboard baseline

  Pattern interpretation              Cluster outputs and engineered
                                      dimensions

  Experiment management               Version registry
  -----------------------------------------------------------------------

## 9. Remaining research-stage work

1.  Obtain supervisor approval for the classification target.
2.  Decide whether additional environmental/geospatial data will be
    integrated.
3.  Validate all cause-code mappings that remain uncertain.
4.  Define the final feature-selection protocol and record it in Chapter
    3.
5.  Implement cluster-profile analysis and latent-pattern
    visualizations.
6.  Run the complete six-year experiment after the target and
    preprocessing rules are approved.
7.  Report results using the same dataset version, preprocessing
    configuration and algorithm parameters recorded by the experiment
    registry.


## v6 Cluster Analysis / Pattern Profiling
- Added `utils/pattern_analysis.py` for descriptive cluster profiling.
- Added cluster summary, numeric/categorical profiles, temporal profiles, cause profiles, and data-quality warnings.
- K-Means K=8 assignments are used for full-dataset profiling.
- Profiling explicitly flags categorical-source inconsistencies and cautions against interpreting clusters substantively before harmonization/sensitivity analysis.

## v7 — Categorical Harmonization and Clustering Sensitivity

A harmonization stage was added before the next clustering baseline. Raw NTSA fields are retained for auditability while canonical fields are created for clustering: `county_h`, `vehicle_type_h`, `gender_h`, and `victim_h`. Vehicle combinations are canonicalized without overwriting the original source value; victim records are grouped into broad victim classes; gender variants are grouped into modelling categories; county spelling variants are mapped conservatively, with unmatched values retained rather than silently discarded.

Sensitivity was run on the full 22,114-row dataset. Because full-data silhouette calculations are computationally expensive, K-Means was fitted on the full population while model-selection metrics were evaluated on the same deterministic 3,000-row evaluation sample for both variants.

| Variant | Selected K | Silhouette (3,000 sample) | CH (sample) | DB (sample) |
|---|---:|---:|---:|---:|
| Harmonized + time_period | 8 | 0.126118 | 175.169 | 2.256022 |
| Harmonized without time_period | 8 | 0.166216 | 198.680 | 2.360634 |

Removing `time_period` increased the sample silhouette and Calinski-Harabasz values but also increased the Davies-Bouldin index. This does not establish that one variant is universally superior. It shows that time_period materially changes the geometry of the clustering problem and should therefore be treated as a sensitivity variable in the research analysis rather than an unquestioned source of latent structure.
