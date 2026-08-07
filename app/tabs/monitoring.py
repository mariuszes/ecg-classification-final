import streamlit as st

from app.settings import GRAFANA_URL, MLFLOW_URL, PROMETHEUS_URL


def render():
    st.title("Monitoring")

    st.write(
        "As the author, I did not try to move these tools into Streamlit, since their own interfaces already provide everything needed."
    )

    st.write(
        "Use MLflow to view saved models and experiment runs, Prometheus to "
        "inspect raw API metrics, and Grafana to view the monitoring dashboard."
    )

    st.markdown(f"- [Grafana]({GRAFANA_URL})")
    st.markdown(f"- [Prometheus]({PROMETHEUS_URL})")
    st.markdown(f"- [MLflow]({MLFLOW_URL})")

    st.markdown(
        """
        API metrics:
        - `ecg_prediction_requests_total`,
        - `ecg_prediction_errors_total`,
        - `ecg_prediction_latency_seconds`,
        - `ecg_prediction_class_total`,
        - `ecg_feature_drift_score`.
        """
    )