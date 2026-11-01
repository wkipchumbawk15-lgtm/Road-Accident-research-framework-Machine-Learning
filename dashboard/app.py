import streamlit as st
import requests
import pandas as pd
import os
import json
import time
from pathlib import Path
from urllib.parse import quote

# API_BASE = os.getenv("API_BASE", "http://localhost:8000")
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="Accident Analytics Dashboard",
    layout="wide"
)

# Compatibility helper: Streamlit removed/changed some experimental APIs across
# versions. Provide a safe rerun helper that attempts the supported approaches
# in order (explicit API -> query param trigger -> session_state toggle).
def _safe_rerun():
    # Try the old experimental rerun if present
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
            return
    except Exception:
        pass

    # Modern API: use st.query_params (read/write dict-like) when available
    try:
        if hasattr(st, "query_params"):
            params = dict(st.query_params)
            params["_rerun"] = str(int(time.time()))
            st.query_params = params
            return
    except Exception:
        pass

    # Fallback for very old Streamlit versions still exposing experimental_get/set
    try:
        params = st.experimental_get_query_params()  # type: ignore[attr-defined]
        params["_rerun"] = str(int(time.time()))
        st.experimental_set_query_params(**params)  # type: ignore[attr-defined]
        return
    except Exception:
        pass

    # Last resort: toggle a session_state key (many widgets will cause a rerun)
    st.session_state["_rerun_flag"] = not st.session_state.get("_rerun_flag", False)

st.title("🚦 Accident Analytics & Prediction Dashboard")
st.caption("Large-scale Road Accident Intelligence System")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.header("System Controls")

# Show current training/evaluation status
try:
    status_resp = requests.get(f"{API_BASE}/evaluation-results", timeout=5).json()
    training_status = status_resp.get("status")
    active_model_status = status_resp.get("active_model")
except Exception:
    training_status = "unavailable"
    active_model_status = None

# If we just activated a model in this session, show a confirmation after reload
if "_just_activated" in st.session_state:
    just_activated = st.session_state.pop("_just_activated", None)
    if just_activated:
        if active_model_status == just_activated:
            st.sidebar.success(f"{just_activated} activated and is now active")
        else:
            st.sidebar.info(f"Activation requested for {just_activated}. Current active model: {active_model_status}")

if training_status == "running":
    st.sidebar.info("Training status: running")
elif training_status == "ready":
    st.sidebar.success(f"Training status: ready — active model: {active_model_status}")
elif training_status == "failed":
    st.sidebar.error("Training status: failed")
elif training_status == "interrupted":
    st.sidebar.warning("Training status: idle — previous run is no longer active")
elif training_status == "unavailable":
    st.sidebar.error("Training status: could not be verified")
else:
    st.sidebar.info("Training status: not started")

if st.sidebar.button("🔍 Load Evaluation Results"):
    r = requests.get(f"{API_BASE}/evaluation-results")
    if r.status_code == 200:
        eval_data = r.json()
        st.session_state["eval"] = eval_data.get("evaluation")
        st.session_state["eval_active_model"] = eval_data.get("active_model")
        st.session_state["eval_status"] = eval_data.get("status")
        st.session_state["current_version"] = eval_data.get("version")
    else:
        st.sidebar.error("Evaluation not available")

# -------------------------------------------------
# DATA UPLOAD
# -------------------------------------------------
st.sidebar.header("📤 Upload Accident Dataset")
uploaded = st.sidebar.file_uploader("Upload NTSA Dataset", type=["csv", "xlsx", "xls"])

# if uploaded:
#     files = {"file": (uploaded.name, uploaded.getvalue())}
#     resp = requests.post(f"{API_BASE}/upload", files=files, timeout=120)
#     st.sidebar.success("File uploaded")
#     st.sidebar.json(resp.json())

if uploaded:
    files = {"file": (uploaded.name, uploaded.getvalue())}

    try:
        resp = requests.post(
            f"{API_BASE}/upload",
            files=files,
            timeout=120
        )

        if resp.status_code == 200:
            upload_info = resp.json()
            st.sidebar.success("File uploaded; preprocessing started")
            st.session_state["uploaded_file"] = upload_info.get("object_name", uploaded.name)
            preprocess_job_id = upload_info.get("job_id")
            if preprocess_job_id:
                for _ in range(60):
                    try:
                        status = requests.get(f"{API_BASE}/preprocess/status/{preprocess_job_id}", timeout=5).json()
                        if status.get("status") == "finished":
                            st.session_state["processed_object"] = status.get("result")
                            st.sidebar.success(f"Preprocessed dataset: {status.get('result')}")
                            break
                        if status.get("failed"):
                            st.sidebar.error("Preprocessing failed")
                            break
                    except Exception:
                        pass
                    time.sleep(1)
            _safe_rerun()

        else:
            st.sidebar.error(f"Upload failed: {resp.text}")

    except Exception as e:
        st.sidebar.error(f"Upload error: {e}")
# -------------------------------------------------
# RESEARCH TARGET
# -------------------------------------------------
if st.session_state.get("processed_object"):
    st.sidebar.header("🎯 Classification Target")
    st.sidebar.caption(
        "The proposal classifies accidents by cause. When the uploaded workbook contains "
        "a Category/Code/Description sheet, its cause categories are used directly; otherwise "
        "the bundled provisional mapping is used. Review the mapping before interpreting results. "
        "DATE is used only to derive temporal predictors, not as a classification target."
    )
    try:
        cols_resp = requests.get(f"{API_BASE}/inspect-columns/{st.session_state['uploaded_file']}", timeout=10).json()
        cols = cols_resp.get("cleaned_columns") or []
        if "cause_category" in cols:
            st.session_state["target_column"] = "cause_category"
            st.sidebar.write("Research classification target: `cause_category`")
        else:
            st.session_state["target_column"] = None
            st.sidebar.error(
                "The processed dataset has no cause_category. Check the cause-code mapping "
                "and preprocess the dataset again before classification."
            )
    except Exception as e:
        st.sidebar.warning(f"Could not inspect processed dataset: {str(e)[:80]}")

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Evaluation", "🧩 Clustering", "🔮 Prediction", "📚 Research Results", "ℹ️ System Info"]
)

# =================================================
# TAB 1: MODEL EVALUATION
# =================================================
with tab1:
    st.subheader("Model Evaluation & Comparison")

    eval_data = st.session_state.get("eval")
    current_version = st.session_state.get("current_version")

    if not eval_data:
        st.info("Click **Load Evaluation Results** from sidebar")
    else:
        if current_version:
            st.info(f"📦 Version: **{current_version}**")
        
        active_model = st.session_state.get("eval_active_model")
        if active_model:
            st.info(f"Active classification model: **{active_model}**")

        raw_df = pd.DataFrame(eval_data).T.reset_index()
        raw_df.rename(columns={"index": "Algorithm"}, inplace=True)

        # Expand nested 'test_metrics' into top-level columns (if present)
        if 'test_metrics' in raw_df.columns:
            metrics_df = pd.json_normalize(raw_df['test_metrics'])
            raw_df = pd.concat([raw_df.drop(columns=['test_metrics']), metrics_df], axis=1)

        # Convert any dict-like columns to JSON strings to avoid Arrow serialization errors
        for col in raw_df.columns:
            if raw_df[col].apply(lambda x: isinstance(x, dict)).any():
                raw_df[col] = raw_df[col].apply(lambda x: json.dumps(x, default=str))

        # Classification summary section
        st.subheader("Classification Summary")
        summary_cols = [c for c in ['Algorithm', 'accuracy', 'precision', 'recall', 'f1_score'] if c in raw_df.columns]
        summary_df = raw_df[summary_cols] if len(summary_cols) > 1 else raw_df
        st.dataframe(summary_df.set_index('Algorithm'), width='stretch')

        st.markdown("---")
        st.subheader("Classification Details")
        st.write("Full model evaluation table and hyperparameters.")
        st.dataframe(raw_df, width='stretch')

        # Choose a metric available in the dataframe for visualization
        available_metrics = [c for c in ['accuracy', 'precision', 'recall', 'f1_score'] if c in raw_df.columns]
        metric = st.selectbox("Select metric to visualize", available_metrics)

        if metric and metric in raw_df.columns:
            chart_df = raw_df[["Algorithm", metric]].set_index("Algorithm")
            st.bar_chart(chart_df)

        st.markdown(
            """
            **Interpretation:**  
            The summary table shows classification performance across all models, while the detailed table includes full metrics and parameter settings.
            """
        )

# =================================================
# TAB 2: CLUSTERING
# =================================================
with tab2:
    st.subheader("Accident Clustering & Pattern Discovery")

    supported_dataset_extensions = {".csv", ".xlsx", ".xls"}
    uploaded_files = []
    try:
        files_response = requests.get(f"{API_BASE}/list-files", timeout=5)
        files_response.raise_for_status()
        listed_files = files_response.json().get("files", [])
        if not isinstance(listed_files, list):
            raise ValueError("The file-list endpoint returned an invalid files list.")

        uploaded_files = sorted({
            filename
            for filename in listed_files
            if isinstance(filename, str)
            and Path(filename).suffix.lower() in supported_dataset_extensions
            and not Path(filename).name.lower().startswith(("cleaned_", "clustering_"))
        })

        recent_upload = st.session_state.get("uploaded_file")
        if (
            isinstance(recent_upload, str)
            and Path(recent_upload).suffix.lower() in supported_dataset_extensions
            and recent_upload not in uploaded_files
        ):
            uploaded_files.append(recent_upload)

    except Exception as e:
        st.warning(f"Could not load uploaded datasets: {e}")

    if uploaded_files:
        recent_upload = st.session_state.get("uploaded_file")
        default_index = (
            uploaded_files.index(recent_upload)
            if recent_upload in uploaded_files
            else 0
        )
        object_name = st.selectbox(
            "Select dataset to cluster:",
            options=uploaded_files,
            index=default_index,
            key="cluster_dataset_selector",
            help="Choose an uploaded CSV or Excel dataset.",
        )
        st.caption(f"Selected dataset: `{object_name}`")
    else:
        st.info(
            "No CSV or Excel datasets are available. Upload a dataset from the "
            "sidebar, then return here to select it."
        )
        object_name = None

    eval_data = st.session_state.get("eval")
    if eval_data is not None:
        classification_df = pd.DataFrame(eval_data).T.reset_index()
        classification_df.rename(columns={"index": "Algorithm"}, inplace=True)
        if 'test_metrics' in classification_df.columns:
            metrics_df = pd.json_normalize(classification_df['test_metrics'])
            classification_df = pd.concat([classification_df.drop(columns=['test_metrics']), metrics_df], axis=1)

        st.markdown("### Classification Summary")
        if 'accuracy' in classification_df.columns:
            summary_cols = [c for c in ['Algorithm', 'accuracy', 'precision', 'recall', 'f1_score'] if c in classification_df.columns]
            st.dataframe(classification_df[summary_cols].set_index('Algorithm'), width='stretch')
        else:
            st.info("Classification summary is not available yet. Load evaluation results first.")

    if st.button("Run Clustering"):
        if not object_name:
            st.error("❌ Please select a dataset first")
        else:
            r = requests.post(
                f"{API_BASE}/run-cluster-background",
                json={"object_name": object_name},
                timeout=30
            )
            if r.status_code == 200:
                data = r.json()
                job_id = data.get("job_id") if isinstance(data, dict) else None
                if job_id:
                    st.info(f"Clustering queued in the background (job_id={job_id}). Polling for completion...")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    for idx in range(120):
                        try:
                            status_resp = requests.get(f"{API_BASE}/cluster-status/{job_id}", timeout=10)
                            if status_resp.status_code == 200:
                                status_data = status_resp.json()
                                status_name = status_data.get("status")
                                if status_name == "finished":
                                    result = status_data.get("result") or {}
                                    if isinstance(result, dict) and result.get("clustering_key"):
                                        clustering_key = result["clustering_key"]
                                        size = result.get("size")
                                        progress_bar.progress(100)
                                        status_text.success(f"Clustering completed and saved to MinIO: {clustering_key}")
                                        st.caption(f"Result size: {size} bytes")
                                        try:
                                            encoded_key = quote(clustering_key, safe="")
                                            presign_resp = requests.get(f"{API_BASE}/presigned-download/{encoded_key}", timeout=10)
                                            if presign_resp.status_code == 200:
                                                download_url_data = presign_resp.json()
                                                url = download_url_data.get("url")
                                                if url:
                                                    st.link_button("Download clustering result", url)
                                                    result_resp = requests.get(url, timeout=30)
                                                    if result_resp.status_code == 200:
                                                        try:
                                                            payload = result_resp.json()
                                                            if isinstance(payload, list) and payload:
                                                                preview_df = pd.DataFrame(payload)
                                                                st.subheader("Clustering result preview")
                                                                st.dataframe(preview_df.head(100), width="stretch")
                                                            elif isinstance(payload, dict):
                                                                st.subheader("Clustering result summary")
                                                                st.json(payload)
                                                            else:
                                                                st.write(payload)
                                                        except Exception:
                                                            st.code(result_resp.text[:2000], language="json")
                                        except Exception:
                                            pass
                                        break
                                    progress_bar.progress(100)
                                    status_text.warning(f"Finished response was returned without a clustering_key: {status_data}")
                                    st.write(status_data)
                                    break
                                if status_name == "failed":
                                    progress_bar.progress(100)
                                    status_text.error(status_data.get("result") or status_data)
                                    break
                                progress = min((idx + 1) / 120 * 100, 99)
                                progress_bar.progress(progress)
                                status_text.info(f"Clustering in progress... {idx + 1}/120 polls")
                        except Exception as e:
                            status_text.warning(f"Polling clustering status failed: {e}")
                        time.sleep(2)
                    else:
                        progress_bar.progress(100)
                        status_text.warning("Clustering is still running or the job was not found yet.")
                else:
                    results = data
                    st.success("Clustering completed")
                    if isinstance(results, list):
                        summary_rows = []
                        for res in results:
                            if isinstance(res, dict) and 'algorithm' in res:
                                quality = res.get('quality_metrics', {})
                                summary_rows.append({
                                    "algorithm": res['algorithm'],
                                    "status": res.get('status'),
                                    "n_clusters": res.get('n_clusters'),
                                    "silhouette_score": quality.get('silhouette_score'),
                                    "calinski_harabasz": quality.get('calinski_harabasz'),
                                    "davies_bouldin": quality.get('davies_bouldin'),
                                    "noise_points": res.get('noise_points')
                                })

                        if summary_rows:
                            st.markdown("### Clustering Summary")
                            st.dataframe(pd.DataFrame(summary_rows).set_index('algorithm'), width='stretch')

                        st.markdown("### Clustering Details")
                        for res in results:
                            if isinstance(res, dict) and 'algorithm' in res:
                                with st.expander(f"{res['algorithm'].replace('_', ' ').title()} details", expanded=True):
                                    cols = st.columns(2)
                                    with cols[0]:
                                        st.write(f"**Status:** {res.get('status')}")
                                        st.write(f"**Clusters:** {res.get('n_clusters')}")
                                        st.write(f"**Cluster Sizes:** {res.get('cluster_sizes')}")
                                        if 'noise_points' in res:
                                            st.write(f"**Noise Points:** {res['noise_points']}")
                                    with cols[1]:
                                        quality = res.get('quality_metrics', {})
                                        if 'silhouette_score' in quality:
                                            st.metric("Silhouette", quality['silhouette_score'])
                                            st.metric("Calinski-Harabasz", quality['calinski_harabasz'])
                                            st.metric("Davies-Bouldin", quality['davies_bouldin'])
                                            st.write("**Interpretation**")
                                            st.write(f"- Silhouette: {quality['interpretation']['silhouette']}")
                                            st.write(f"- Calinski: {quality['interpretation']['calinski']}")
                                            st.write(f"- Davies: {quality['interpretation']['davies']}")
                                        else:
                                            st.write(quality.get('note', 'No quality metrics available'))
                            else:
                                st.write(f"Unexpected result format: {res}")
                    else:
                        st.write(f"Unexpected response format: {results}")
            else:
                st.error(f"Clustering failed: {r.text}")

    st.info(
        """
        **Clustering Use Case:**  
        Groups accidents based on similarity (location, time, conditions),
        revealing high-risk zones and recurring accident patterns.
        """
    )

    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/KMeans-example.png/640px-KMeans-example.png",
        caption="Example of K-Means spatial clustering"
    )

# =================================================
# TAB 3: PREDICTION
# =================================================
with tab3:
    st.subheader("Accident Cause-Category Prediction")

    try:
        meta = requests.get(f"{API_BASE}/metadata", timeout=5).json()
        features = meta["features"]
        algorithm = meta["algorithm"]
        feature_types = meta.get("feature_types", {})
        feature_options = meta.get("feature_options", {})
        feature_defaults = meta.get("feature_defaults", {})

        st.info(f"Active Model: **{algorithm}**")
    except Exception as e:
        st.error(f"Could not load prediction service: {str(e)[:100]}")
        st.info("Make sure the API gateway is running: `docker-compose up -d`")
        features = []
        algorithm = None
        feature_types = {}
        feature_options = {}
        feature_defaults = {}

    if not features or not algorithm:
        st.warning("Prediction service unavailable. Check that the API is running.")

    with st.form("predict_form"):
        st.markdown("### Input Features")
        st.caption(
            "Enter accident characteristics to estimate a mapped cause category. "
            "This predicts recorded cause labels; it does not estimate accident severity or prove causation."
        )

        inputs = {}
        cols = st.columns(3)

        for i, f in enumerate(features):
            with cols[i % 3]:
                label = f.replace("_", " ").title()
                default = feature_defaults.get(f)
                if feature_types.get(f) == "numeric":
                    inputs[f] = st.number_input(
                        label,
                        value=float(default) if default is not None else 0.0,
                    )
                elif f in {"road", "place"}:
                    inputs[f] = st.text_input(label, value=str(default or ""))
                else:
                    options = feature_options.get(f, [])
                    if options:
                        default_value = str(default) if default is not None else options[0]
                        default_index = options.index(default_value) if default_value in options else 0
                        inputs[f] = st.selectbox(label, options, index=default_index)
                    else:
                        inputs[f] = st.text_input(label, value=str(default or ""))

        submit = st.form_submit_button("Predict Cause Category")

    if submit:
        payload = {"data": inputs}
        r = requests.post(f"{API_BASE}/predict", json=payload)

        if r.status_code == 200:
            res = r.json()

            st.success("Cause-category prediction completed")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted Cause Category",
                    res["prediction"]
                )

            with col2:
                st.write("Model:", res["algorithm"])

            # ----------------------------
            # Probability Visualization
            # ----------------------------
            if res.get("probabilities"):
                class_labels = res.get("class_labels", [])
                if len(class_labels) == len(res["probabilities"]):
                    probs = pd.DataFrame(
                        {"Cause category": class_labels, "Probability": res["probabilities"]}
                    ).set_index("Cause category")
                    st.subheader("Predicted Class Probabilities")
                    st.bar_chart(probs)

            # ----------------------------
            # Feature Transparency
            # ----------------------------
            st.subheader("Input Feature Summary")
            st.table(pd.DataFrame.from_dict(
                inputs, orient="index", columns=["Value"]
            ))

        else:
            st.error(r.text)

# =================================================
# TAB 4: RESEARCH RESULTS / DECISION SUPPORT
# =================================================
with tab4:
    st.subheader("📚 NTSA Research Results & Decision Support")
    st.caption("Research evidence generated from the NTSA 2018–2023 implementation.")

    data_dir = Path("/data")
    manifest_path = data_dir / "research_results_manifest.json"
    dataset_path = data_dir / "NTSA_2018_2023_feature_engineered.csv"

    if manifest_path.is_file():
        try:
            research = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            research = {}
            st.error(f"Research results manifest could not be read: {exc}")
    else:
        research = {}
        st.info(
            "Research results have not been generated yet. From the "
            "`accident-system` directory, run "
            "`python -m utils.build_research_artifacts`."
        )

    if dataset_path.is_file():
        try:
            rdf = pd.read_csv(dataset_path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            rdf = pd.DataFrame()
            st.warning(f"Feature-engineered dataset could not be read: {exc}")
    else:
        rdf = pd.DataFrame()
        st.info(
            "The feature-engineered dataset is missing. Generate it with "
            "`python -m utils.build_research_artifacts` from the `accident-system` directory."
        )

    if research:
        d=research.get("dataset",{})
        q=research.get("data_quality",{})

        c1,c2,c3,c4=st.columns(4)
        c1.metric("NTSA Records",f"{d.get('records',0):,}")
        c2.metric("Date Recovery",f"{q.get('date_recovered',0):,}")
        c3.metric("Invalid/Missing Age",f"{q.get('age_missing_or_invalid',0):,}")
        c4.metric("Mapped Cause Descriptions",f"{q.get('mapped_cause_descriptions',0):,}")

        rtab1,rtab2,rtab3,rtab4=st.tabs(
            ["Clustering","Classification","Feature Importance","Data Quality"]
        )

        with rtab1:
            st.markdown("### Clustering model comparison")
            cdf=pd.DataFrame(research.get("clustering",[]))
            if cdf.empty:
                st.info(
                    "No clustering experiments have been added to this research manifest. "
                    "Interactive clustering runs are separate from the saved research comparison."
                )
            else:
                display_columns = [
                    "algorithm",
                    "n_clusters",
                    "noise_count",
                    "records",
                    "silhouette",
                    "calinski_harabasz",
                    "davies_bouldin",
                ]
                clustering_display = cdf.reindex(columns=display_columns).rename(
                    columns={
                        "algorithm": "Algorithm",
                        "n_clusters": "Clusters",
                        "noise_count": "Noise",
                        "records": "Rows",
                        "silhouette": "Silhouette",
                        "calinski_harabasz": "Calinski-Harabasz",
                        "davies_bouldin": "Davies-Bouldin",
                    }
                )
                for column in [
                    "Silhouette",
                    "Calinski-Harabasz",
                    "Davies-Bouldin",
                ]:
                    clustering_display[column] = clustering_display[column].map(
                        lambda value: f"{value:.4f}" if pd.notna(value) else "N/A"
                    )
                for column in ["Noise", "Rows"]:
                    clustering_display[column] = clustering_display[column].map(
                        lambda value: f"{int(value):,}" if pd.notna(value) else "N/A"
                    )
                st.dataframe(
                    clustering_display,
                    use_container_width=True,
                    hide_index=True,
                )
                if {"algorithm", "silhouette"} <= set(cdf.columns):
                    chart_df = cdf.dropna(subset=["silhouette"])
                else:
                    chart_df = pd.DataFrame()
                if not chart_df.empty:
                    st.plotly_chart(
                        __import__("plotly.express",fromlist=["bar"]).bar(
                            chart_df,x="algorithm",y="silhouette",
                            title="Clustering Silhouette Scores"
                        ),
                        use_container_width=True
                    )
                dbscan_rows = cdf[
                    cdf["algorithm"].astype(str).str.lower().eq("dbscan")
                ] if "algorithm" in cdf.columns else pd.DataFrame()
                if not dbscan_rows.empty and {
                    "noise_count", "records"
                } <= set(dbscan_rows.columns):
                    dbscan = dbscan_rows.iloc[0]
                    if dbscan["records"]:
                        noise_pct = 100 * dbscan["noise_count"] / dbscan["records"]
                        if dbscan["n_clusters"] == 0:
                            st.markdown(
                                f"**Interpretation:** DBSCAN classified all records "
                                f"as noise ({noise_pct:.2f}%); no valid cluster-quality "
                                "score is available."
                            )
                        elif dbscan["silhouette"] is not None:
                            st.markdown(
                                f"**Interpretation:** DBSCAN marked {noise_pct:.2f}% "
                                "of records as noise. Do not select it solely because "
                                "of a high silhouette score."
                            )
            sdf=pd.DataFrame(research.get("clustering_sensitivity",[]))
            if not sdf.empty:
                st.markdown("### Sensitivity to time-period representation")
                st.dataframe(sdf,use_container_width=True)

        with rtab2:
            st.markdown("### Classification model comparison")
            clf=pd.DataFrame(research.get("classification",[]))
            if clf.empty:
                st.info("No matching saved classification evaluation is available.")
            else:
                display_columns = [
                    "model",
                    "target",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "cv_f1_weighted",
                ]
                classification_display = clf.reindex(
                    columns=display_columns
                ).rename(
                    columns={
                        "model": "Algorithm",
                        "target": "Target",
                        "accuracy": "Accuracy",
                        "precision": "Weighted precision",
                        "recall": "Weighted recall",
                        "f1_score": "Weighted F1",
                        "cv_f1_weighted": "CV weighted F1",
                    }
                )
                for column in [
                    "Accuracy",
                    "Weighted precision",
                    "Weighted recall",
                    "Weighted F1",
                    "CV weighted F1",
                ]:
                    classification_display[column] = classification_display[
                        column
                    ].map(
                        lambda value: f"{value:.4f}"
                        if pd.notna(value)
                        else "N/A"
                    )
                st.dataframe(
                    classification_display,
                    use_container_width=True,
                    hide_index=True,
                )
            if not clf.empty:
                st.plotly_chart(
                    __import__("plotly.express",fromlist=["bar"]).bar(
                        clf,x="model",y="accuracy",color="target",
                        barmode="group",title="Classification Accuracy"
                    ),
                    use_container_width=True
                )
            st.caption(
                "The broader cause-category target is researcher-derived from mapped descriptions; "
                "it is not an official NTSA taxonomy."
            )

        with rtab3:
            st.markdown("### Random Forest feature importance")
            fidf=pd.DataFrame(research.get("feature_importance",[]))
            if not fidf.empty:
                st.dataframe(fidf,use_container_width=True)
                st.plotly_chart(
                    __import__("plotly.express",fromlist=["bar"]).bar(
                        fidf.sort_values("importance"),
                        x="importance",y="feature",orientation="h",
                        title="Top Random Forest Features"
                    ),
                    use_container_width=True
                )
            else:
                st.info(
                    "Feature-importance analysis has not been generated for this manifest."
                )

        with rtab4:
            qdf=pd.DataFrame([
                ["Original DATE missing",q.get("date_missing",0)],
                ["Recovered continuation dates",q.get("date_recovered",0)],
                ["DATE/DAY conflicts",q.get("date_day_conflicts",0)],
                ["TIME missing",q.get("time_missing",0)],
                ["AGE missing/invalid",q.get("age_missing_or_invalid",0)],
                ["CAUSE CODE missing",q.get("cause_code_missing",0)],
                ["Mapped cause descriptions",q.get("mapped_cause_descriptions",0)]
            ],columns=["Quality indicator","Records"])
            st.dataframe(qdf,use_container_width=True)
            st.plotly_chart(
                __import__("plotly.express",fromlist=["bar"]).bar(
                    qdf,x="Quality indicator",y="Records",
                    title="NTSA Data Quality Indicators"
                ),
                use_container_width=True
            )

    if not rdf.empty:
        st.markdown("### Interactive accident pattern exploration")
        f1,f2=st.columns(2)
        with f1:
            if "year" in rdf:
                y=rdf.groupby("year").size().reset_index(name="records")
                st.plotly_chart(
                    __import__("plotly.express",fromlist=["line"]).line(
                        y,x="year",y="records",markers=True,
                        title="Records by Year"
                    ),
                    use_container_width=True
                )
        with f2:
            if "county" in rdf:
                c=rdf["county"].value_counts().head(15).reset_index()
                c.columns=["county","records"]
                st.plotly_chart(
                    __import__("plotly.express",fromlist=["bar"]).bar(
                        c,x="county",y="records",
                        title="Top Counties"
                    ),
                    use_container_width=True
                )

    st.markdown("### Research interpretation")
    st.markdown("""
- County and vehicle type are among the strongest Random Forest predictors.
- Temporal representation has a measurable effect on cluster structure.
- Exact cause-code classification is difficult because the target contains many imbalanced classes.
- Broader researcher-derived cause categories are easier to classify, but remain imbalanced.
- The dashboard reports associations and patterns; it does not establish causal relationships.
""")

# =================================================
# TAB 5: SYSTEM INFO
# =================================================
with tab5:
    st.subheader("System Architecture Overview")

    st.markdown(
        """
        **Backend:**  
        - FastAPI Microservices  
        - ML models served via REST  
        - Docker & Kubernetes ready  

        **Frontend:**  
        - Streamlit dashboard  
        - Dynamic UI (no hardcoding)  

        **Scalability:**  
        - Supports batch processing  
        - Message queue ready  
        - Cloud deployable  

        **Research Value:**  
        - Cause-category pattern prediction for recorded accident cases
        - Supports policy and safety interventions  
        """
    )

st.sidebar.header("🚀 Quick Workflow")

if st.sidebar.button("Run Full Analysis Workflow"):
    st.sidebar.info("Starting full workflow: Upload → Train → Evaluate → Select Best → Activate")
    
    # Step 1: Check if data is uploaded (assume user has uploaded via uploader)
    # Note: Actual upload is manual, but we can check if eval data exists
    
    # Step 2: Train models (use current selections)
    selected_classifiers = st.session_state.get("selected_classifiers", ["random_forest", "svm"])
    selected_clusterers = st.session_state.get("selected_clusterers", ["kmeans", "dbscan", "agglomerative", "mean_shift"])
    processed_object = st.session_state.get("processed_object")
    target_column = st.session_state.get("target_column")

    if not processed_object:
        st.sidebar.error("Upload and preprocess a research dataset first.")
    elif not target_column:
        st.sidebar.error("Set a research classification target before training.")
    elif not selected_classifiers and not selected_clusterers:
        st.sidebar.error("Select algorithms first")
    else:
        try:
            payload = {
                "dataset_object": processed_object,
                "target_column": target_column,
                "classifiers": selected_classifiers,
                "clusterers": selected_clusterers
            }
            r = requests.post(f"{API_BASE}/train-models", json=payload, timeout=10)
            if r.status_code >= 400:
                st.sidebar.error(f"Training failed: {r.text}")
            else:
                data = r.json()
                job_id = data.get("job_id")
                if job_id:
                    st.sidebar.info("Training started... (this may take time)")
                    # Poll for completion
                    timeout = 3600
                    poll_interval = 5
                    elapsed = 0
                    with st.spinner("Workflow in progress: Training → Evaluating..."):
                        while elapsed < timeout:
                            try:
                                resp = requests.get(f"{API_BASE}/train-status/{job_id}", timeout=10)
                                if resp.status_code == 200:
                                    js = resp.json()
                                    status = js.get("status")
                                    if status == "finished" or js.get("is_finished"):
                                        result = js.get("result")
                                        if isinstance(result, dict) and result.get("status") == "skipped":
                                            st.sidebar.warning(
                                                "Training was skipped: "
                                                f"{result.get('message', 'another run is in progress.')}"
                                            )
                                            break
                                        if not isinstance(result, dict) or result.get("status") != "ready":
                                            st.sidebar.error(
                                                "Training job finished without a successful training result."
                                            )
                                            break
                                        st.sidebar.success("Training completed!")
                                        # Load results
                                        eval_resp = requests.get(f"{API_BASE}/evaluation-results", timeout=10)
                                        if eval_resp.status_code == 200:
                                            st.session_state["eval"] = eval_resp.json().get("evaluation")
                                            st.success("✅ Workflow Complete: Models trained, evaluated, and ready!")
                                            # Auto-select best model
                                            eval_data = st.session_state["eval"]
                                            if eval_data:
                                                # Find best classifier by accuracy
                                                best_algo = max(eval_data.keys(), key=lambda k: eval_data[k].get('accuracy', 0))
                                                st.info(f"🔥 Best model selected: {best_algo}")
                                                # Auto-activate
                                                try:
                                                    resp = requests.post(f"{API_BASE}/models/select", json={"model_name": best_algo})
                                                    if resp.status_code == 200:
                                                        st.success(f"✅ {best_algo} activated for predictions!")
                                                    else:
                                                        st.warning("Could not auto-activate model")
                                                except Exception:
                                                    st.warning("Could not auto-activate model")
                                        break
                                    elif status == "failed" or js.get("is_failed"):
                                        st.sidebar.error(
                                            f"Training failed: {js.get('error') or js.get('result') or 'No error details were returned.'}"
                                        )
                                        break
                            except Exception:
                                pass
                            time.sleep(poll_interval)
                            elapsed += poll_interval
                        else:
                            st.sidebar.warning("Workflow timed out")
                else:
                    st.sidebar.error("Training did not start properly")
        except Exception as e:
            st.sidebar.error(f"Workflow error: {e}")

st.sidebar.markdown("---")
try:
    versions_resp = requests.get(f"{API_BASE}/versions", timeout=5).json()
    versions_data = versions_resp.get("versions", [])
    
    # Extract latest scores for preview
    latest_scores = {}
    if versions_data:
        latest_version = versions_data[-1]  # Assuming versions_data is sorted oldest to newest
        latest_scores = latest_version.get("classification_scores", {})
except Exception as e:
    st.sidebar.warning(f"Could not load versions for preview: {str(e)[:50]}")
    versions_data = []
    latest_scores = {}

# Algorithm selection
st.sidebar.subheader("Select Algorithms to Train")

# Classification algorithms
st.sidebar.write("**Classification Algorithms:**")
classification_algorithms = ["random_forest", "svm"]
selected_classifiers = []
for algo in classification_algorithms:
    score = latest_scores.get(algo, {}).get('accuracy', 'N/A')
    if st.sidebar.checkbox(f"{algo.replace('_', ' ').title()} (Latest Acc: {score})", value=True, key=f"clf_{algo}"):
        selected_classifiers.append(algo)

# Clustering algorithms
st.sidebar.write("**Clustering Algorithms:**")
clustering_algorithms = ["kmeans", "dbscan", "agglomerative", "mean_shift"]
selected_clusterers = []
for algo in clustering_algorithms:
    if st.sidebar.checkbox(algo.replace("_", " ").title(), value=True, key=f"clust_{algo}"):
        selected_clusterers.append(algo)

# Store selections in session state
st.session_state["selected_classifiers"] = selected_classifiers
st.session_state["selected_clusterers"] = selected_clusterers

# Display selected algorithms
if selected_classifiers or selected_clusterers:
    st.sidebar.success(
        f"Training: {len(selected_classifiers)} classifiers, {len(selected_clusterers)} clusterers"
    )

# =========================================================
# Version History Section
# =========================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Versions")

if versions_data:
    # Display version history as expandable cards
    for version_info in reversed(versions_data):  # Show newest first
        version_name = version_info.get("version")
        timestamp = version_info.get("timestamp", "Unknown")
        best_classifier = version_info.get("best_classifier", "N/A")
        classification_scores = version_info.get("classification_scores", {})
        
        # Format timestamp
        try:
            from datetime import datetime
            ts = datetime.fromisoformat(timestamp)
            ts_str = ts.strftime("%Y-%m-%d %H:%M")
        except:
            ts_str = timestamp
        
        # Create expandable section for version
        with st.sidebar.expander(f"🏷️ {version_name} ({ts_str})"):
            st.write(f"**Best Classifier:** {best_classifier}")
            
            # Show best classifier's scores
            if best_classifier in classification_scores:
                scores = classification_scores[best_classifier]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{scores.get('accuracy', 'N/A')}")
                    st.metric("F1-Score", f"{scores.get('f1_score', 'N/A')}")
                with col2:
                    st.metric("Precision", f"{scores.get('precision', 'N/A')}")
                    st.metric("Recall", f"{scores.get('recall', 'N/A')}")
            
            # Show all classifiers for this version
            st.write("**All Classifiers:**")
            for clf_name, scores in classification_scores.items():
                st.caption(f"{clf_name}: acc={scores.get('accuracy')}, f1={scores.get('f1_score')}")
            
            # Activate version button
            if st.button(f"Activate {version_name}", key=f"activate_{version_name}"):
                try:
                    resp = requests.post(f"{API_BASE}/versions/activate/{version_name}")
                    if resp.status_code == 200:
                        st.sidebar.success(f"{version_name} activated")
                        st.session_state["_version_activated"] = version_name
                    else:
                        st.sidebar.error(f"Failed to activate {version_name}")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
else:
    st.sidebar.info("No versions yet. Train models to create v1.")

# =========================================================
# Model Selection (keeps existing functionality)
# =========================================================
st.sidebar.markdown("---")

st.sidebar.header("Active Model Selection")

# Fetch models safely
try:
    models_resp = requests.get(f"{API_BASE}/models", timeout=5).json()
except Exception as e:
    st.sidebar.error("Failed to fetch models from API")
    models_resp = {"models": [], "active_model": None}

models = models_resp.get("models", [])
active_model = models_resp.get("active_model")

if not models:
    st.sidebar.info("No models available. Train models first.")
else:
    # Determine selected index safely
    try:
        selected_index = models.index(active_model) if active_model in models else 0
    except Exception:
        selected_index = 0

    selected = st.sidebar.selectbox(
        "Select active model",
        models,
        index=selected_index
    )

    if st.sidebar.button("Activate Model"):
        resp = requests.post(f"{API_BASE}/models/select", json={"model_name": selected})
        if resp.status_code == 200:
            # Persist activation intent in session so the reload can confirm it
            st.session_state["_just_activated"] = selected
            _safe_rerun()
        else:
            st.error("Failed to activate model")

    if st.sidebar.button("Delete Selected Model"):
        st.session_state["_pending_model_deletion"] = selected

    pending_model_deletion = st.session_state.get("_pending_model_deletion")
    if pending_model_deletion:
        st.sidebar.warning(
            f"Delete trained model '{pending_model_deletion}'? "
            + ("This is the active model and will be deactivated. " if pending_model_deletion == active_model else "")
            + "This action cannot be undone."
        )
        confirm_col, cancel_col = st.sidebar.columns(2)
        if confirm_col.button("Confirm Delete", key="confirm_model_deletion"):
            try:
                response = requests.delete(
                    f"{API_BASE}/models/{quote(pending_model_deletion, safe='')}",
                    timeout=10
                )
            except requests.RequestException as exc:
                st.sidebar.error(f"Failed to delete model: {exc}")
            else:
                if response.status_code == 200:
                    st.session_state.pop("_pending_model_deletion", None)
                    st.sidebar.success(f"Model '{pending_model_deletion}' deleted")
                    _safe_rerun()
                else:
                    st.sidebar.error(f"Failed to delete model: {response.text}")
        if cancel_col.button("Cancel", key="cancel_model_deletion"):
            st.session_state.pop("_pending_model_deletion", None)
            _safe_rerun()

if st.sidebar.button("Train Models"):
    selected_classifiers = st.session_state.get("selected_classifiers", ["random_forest", "svm"])
    selected_clusterers = st.session_state.get("selected_clusterers", ["kmeans", "dbscan", "agglomerative", "mean_shift"])
    processed_object = st.session_state.get("processed_object")
    target_column = st.session_state.get("target_column")

    if not processed_object:
        st.sidebar.error("Upload and preprocess a research dataset first.")
    elif not target_column:
        st.sidebar.error("Set a research classification target before training.")
    elif not selected_classifiers and not selected_clusterers:
        st.sidebar.warning("Select at least one algorithm to train")
    else:
        try:
            payload = {
                "dataset_object": processed_object,
                "target_column": target_column,
                "classifiers": selected_classifiers,
                "clusterers": selected_clusterers
            }
            r = requests.post(f"{API_BASE}/train-models", json=payload, timeout=10)
        except Exception as e:
            st.sidebar.error(f"Failed to start training: {e}")
        else:
            if r.status_code >= 400:
                st.sidebar.error(f"Training request failed: {r.text}")
            else:
                data = r.json()
                job_id = data.get("job_id")
                if job_id:
                    st.sidebar.info(f"Training queued (job_id={job_id}) — polling job status...")
                    timeout = 3600
                    poll_interval = 3
                    elapsed = 0
                    with st.spinner("Training in progress — waiting for results..."):
                        while elapsed < timeout:
                            try:
                                resp = requests.get(f"{API_BASE}/train-status/{job_id}", timeout=10)
                                if resp.status_code == 200:
                                    js = resp.json()
                                    status = js.get("status")
                                    if status == "finished" or js.get("is_finished"):
                                        result = js.get("result")
                                        if isinstance(result, dict) and result.get("status") == "skipped":
                                            st.sidebar.warning(
                                                "Training was skipped: "
                                                f"{result.get('message', 'another run is in progress.')}"
                                            )
                                            break
                                        if not isinstance(result, dict) or result.get("status") != "ready":
                                            st.sidebar.error(
                                                "Training job finished without a successful training result."
                                            )
                                            break
                                        # Load evaluation results after job finishes
                                        eval_resp = requests.get(f"{API_BASE}/evaluation-results", timeout=10)
                                        if eval_resp.status_code == 200:
                                            st.sidebar.success("Training completed")
                                            st.session_state["eval"] = eval_resp.json().get("evaluation")
                                            st.success("Training completed — evaluation loaded")
                                            break
                                    elif status == "failed" or js.get("is_failed"):
                                        st.sidebar.error(
                                            f"Training job failed: {js.get('error') or js.get('result') or 'No error details were returned.'}"
                                        )
                                        break
                            except Exception:
                                pass

                            time.sleep(poll_interval)
                            elapsed += poll_interval

                        else:
                            st.sidebar.warning("Training is still running (timeout reached)")

                    if st.session_state.get("eval"):
                        _safe_rerun()

                else:
                    # Fallback: previous behavior (poll evaluation file)
                    st.sidebar.info("Training started — polling for results...")
                    timeout = 300  # seconds
                    poll_interval = 3
                    elapsed = 0
                    with st.spinner("Training in progress — waiting for results..."):
                        while elapsed < timeout:
                            try:
                                resp = requests.get(f"{API_BASE}/evaluation-results", timeout=10)
                                if resp.status_code == 200:
                                    data = resp.json()
                                    status = data.get("status")
                                    if status == "ready":
                                        st.sidebar.success("Training completed")
                                        st.session_state["eval"] = data.get("evaluation")
                                        st.success("Training completed — evaluation loaded")
                                        break
                                    elif status == "failed":
                                        st.sidebar.error(f"Training failed: {data.get('message')}")
                                        break
                            except Exception:
                                pass

                            time.sleep(poll_interval)
                            elapsed += poll_interval

                        else:
                            st.sidebar.warning("Training is still running (timeout reached)")

                    # Refresh eval_data displayed in the dashboard if available
                    if st.session_state.get("eval"):
                        _safe_rerun()
