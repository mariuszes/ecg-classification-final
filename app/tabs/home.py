import streamlit as st


def render():
    st.title("ECG Classification and Monitoring")

    st.write(
        "This application demonstrates an end-to-end MLOps workflow for binary ECG "
        "classification with MLflow, FastAPI, Prometheus, Grafana and Streamlit."
    )

    st.subheader("Application flow")
    st.code(
        "PTB-XL+ → ML pipeline → MLflow champion → FastAPI → Streamlit",
        language=None,
    )

    prediction_col, drift_col, monitoring_col = st.columns(3)

    with prediction_col:
        st.markdown("### Prediction")
        st.write(
            "Select a row from `test.csv` and compare the champion model "
            "prediction with the true label."
        )

    with drift_col:
        st.markdown("### Drift simulation")
        st.write(
            "Create normal, noisy or shifted batches and observe how the data "
            "change affects predictions."
        )

    with monitoring_col:
        st.markdown("### Monitoring")
        st.write(
            "Track requests, errors, latency, class distribution and drift "
            "with Prometheus and Grafana."
        )

    st.subheader("Feature drift")

    st.markdown(
        """
        The training dataset is used as the reference profile.

        * **normal** - the selected batch is used without modification.
        * **noisy** - the same change is additionally multiplied by a random
        noise value generated separately for every cell:
        """
    )

    st.latex(
        r"x'_{i,j} = x_{i,j} + "
        r"z_{i,j} \cdot \sigma_{\mathrm{train},j} \cdot s"
    )

    st.markdown(
        """
        * **drifted** - every value in feature $j$ is shifted by its training
        standard deviation multiplied by the selected strength:
        """
    )

    st.latex(r"x'_{i,j} = x_{i,j} + \sigma_{\mathrm{train},j} \cdot s")

    st.markdown("**Where:**")

    st.latex(
        r"""
        \begin{aligned}
            i & - \text{index of the currently evaluated row in the batch}, \\
            j & - \text{index of the currently evaluated feature}, \\
            x_{i,j} & - \text{original value of feature } j
            \text{ in row } i, \\
            x'_{i,j} & - \text{modified value of feature } j
            \text{ in row } i, \\
            \sigma_{\mathrm{train},j} & - \text{standard deviation of feature } j
            \text{ in the training dataset}, \\
            s & - \text{change strength selected in the Drift simulation tab}, \\
            z_{i,j} & - \text{Gaussian noise sampled separately for each cell} \\
                & \quad \text{from the standard normal distribution } \mathcal{N}(0,1).
        \end{aligned}
        """
    )

    st.write(
        "N(0,1) means a normal distribution with mean 0 and standard deviation 1. "
        "About 68% of noise values are between -1 and 1. Values outside this range "
        "can still occur, but values below -3 or above 3 are very rare, with a total "
        "probability of about 0.3%."
    )

    st.subheader("Drift score")

    st.write(
        "The formula below shows how the drift score is calculated for the selected batch."
    )

    st.latex(
        r"\mathrm{drift\ score}="
        r"\frac{1}{N}\sum_{j=1}^{N}"
        r"\frac{\left|\mu_{\mathrm{batch},j}-\mu_{\mathrm{train},j}\right|}"
        r"{\sigma_{\mathrm{train},j}}"
    )


    st.markdown("**Where:**")

    st.latex(
        r"""
        \begin{aligned}
            j & - \text{index of the currently evaluated feature}, \\
            N & - \text{total number of model features in the batch}, \\
            \mu_{\mathrm{batch},j} & - \text{mean value of feature } j
            \text{ in the selected batch}, \\
            \mu_{\mathrm{train},j} & - \text{mean value of feature } j
            \text{ in the training dataset}, \\
            \sigma_{\mathrm{train},j} & - \text{standard deviation of feature } j
            \text{ in the training dataset}.
        \end{aligned}
        """
    )


    st.caption(
        "The score is the average difference between batch and training feature "
        "means, expressed in training standard deviations."
    )

    st.subheader("Available tabs")
    st.markdown(
        """
        * **Prediction demo** - single-row prediction through `/predict-row`
        * **Drift simulation** - batch predictions through `/predict` and drift reporting through `/simulation-metrics`
        * **Monitoring** - direct links to MLflow, Prometheus and Grafana
        """
    )
