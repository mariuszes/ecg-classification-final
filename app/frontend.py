import streamlit as st

from app.tabs import drift, home, monitoring, prediction

st.set_page_config(page_title="ECG Monitoring Demo", layout="wide")

pages = st.tabs(["Home", "Prediction demo", "Drift simulation", "Monitoring"])

with pages[0]:
    home.render()
with pages[1]:
    prediction.render()
with pages[2]:
    drift.render()
with pages[3]:
    monitoring.render()
