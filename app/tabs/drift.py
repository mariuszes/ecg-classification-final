import time

import pandas as pd
import requests
import streamlit as st

from app.settings import API_URL, TEST_PATH
from src.drift import modify_batch, reference_profile, simple_drift_score


@st.cache_data
def load_test_data():
    return pd.read_csv(TEST_PATH)


def render():
    st.title("Drift simulation")
    st.write("Send a batch of 12SL features to the API and see how changes in the data affect the model's predictions.")

    scenario = st.selectbox("Scenario", ["normal", "noisy", "drifted"])
    defaults = {"normal": 0.0, "noisy": 0.15, "drifted": 0.5}
    strength = st.slider(
        "Change strength",
        min_value=0.0,
        max_value=1.0,
        value=defaults[scenario],
        step=0.05,
        disabled=scenario == "normal",
        key=f"strength_{scenario}",
    )
    batch_size = st.slider(
        "Batch size",
        min_value=50,
        max_value=500,
        value=100,
        step=50,
    )
    delay = st.slider("Delay between requests (seconds)", 0.0, 1.0, 0.5, 0.1)

    if st.button("Run simulation", type="primary"):
        test_df = load_test_data()
        reference_mean, reference_std = reference_profile()
        batch = test_df.sample(n=batch_size, random_state=42).reset_index(drop=True)
        batch = modify_batch(batch, reference_std, scenario, strength)
        drift_score = simple_drift_score(batch, reference_mean, reference_std)

        try:
            requests.post(
                f"{API_URL}/simulation-metrics",
                json={"scenario": scenario, "drift_score": drift_score},
                timeout=30,
            ).raise_for_status()
        except requests.RequestException as exc:
            st.error(f"API is not available: {exc}")
            return

        results = []
        errors = 0
        progress = st.progress(0)
        feature_names = list(reference_mean.index)

        for index, row in batch.iterrows():
            features = {name: float(row[name]) for name in feature_names}
            start = time.perf_counter()

            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"features": features, "scenario": scenario},
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()
                results.append(
                    {
                        "request": index + 1,
                        "prediction": result["prediction_label"],
                        "probability_abnormal": result["probability_abnormal"],
                        "latency_ms": (time.perf_counter() - start) * 1000,
                    }
                )
            except requests.RequestException:
                errors += 1

            progress.progress((index + 1) / batch_size)
            if delay:
                time.sleep(delay)

        result_df = pd.DataFrame(results)
        st.subheader("Batch summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Requests sent", batch_size)
        col2.metric("Errors", errors)
        col3.metric("Drift score", f"{drift_score:.4f}")

        if result_df.empty:
            return

        col1, col2, col3 = st.columns(3)
        col1.metric("Average latency", f"{result_df['latency_ms'].mean():.1f} ms")
        col2.metric("Predicted abnormal", f"{(result_df['prediction'] == 'abnormal').mean() * 100:.1f}%")
        col3.metric("Average probability abnormal", f"{result_df['probability_abnormal'].mean():.3f}")

        st.subheader("Probability abnormal by request")
        st.line_chart(result_df.set_index("request")["probability_abnormal"])
        st.subheader("Response time by request")
        st.line_chart(result_df.set_index("request")["latency_ms"])
        st.dataframe(result_df, hide_index=True, use_container_width=True)
