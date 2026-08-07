import pandas as pd
import requests
import streamlit as st

from app.settings import API_URL, TEST_PATH


def render():
    st.title("Prediction demo")

    if not TEST_PATH.exists():
        st.error("Missing test.csv. Run the data pipeline first.")
        return

    df = pd.read_csv(TEST_PATH)

    st.write("Browse `test.csv`, choose a row and run the champion model.")
    st.dataframe(df, height=350, use_container_width=True)

    row_index = st.number_input(
        "Row index",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1,
    )

    if not st.button("Predict", type="primary"):
        return

    try:
        response = requests.post(
            f"{API_URL}/predict-row",
            json={"row_index": int(row_index)},
            timeout=30,
        )

        if response.status_code != 200:
            st.error(response.json().get("detail", response.text))
            return

        result = response.json()

        left, right = st.columns(2)
        left.metric("Prediction", result["prediction_label"])
        right.metric("True label", result["true_label"])

        if result.get("probability_abnormal") is not None:
            st.metric(
                "Abnormal probability",
                f"{result['probability_abnormal']:.3f}",
            )

    except requests.RequestException as error:
        st.error(f"API is not available: {error}")