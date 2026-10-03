import os
import subprocess
import sys

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dhaka House Rent Predictor", page_icon="🏠")
st.title("🏠 Dhaka House Rent Predictor")
st.write("Flat er details dao, model mashik bhara estimate korbe.")


@st.cache_resource
def load_bundle():
    # model.joblib na thakle train.py chalie nijei toiri kore nebe
    if not os.path.exists("model.joblib"):
        with st.spinner("Prothom bar model train hocche, 1-2 minute lagbe..."):
            subprocess.run([sys.executable, "train.py"], check=True)
    return joblib.load("model.joblib")


bundle = load_bundle()
model = bundle["model"]

row = {}
labels = {"Area": "Size (sqft)", "Bed": "Bedrooms", "Bath": "Bathrooms"}
for col, (lo, med, hi) in bundle["numeric"].items():
    row[col] = st.number_input(labels.get(col, col), min_value=lo, max_value=hi, value=med)
for col, options in bundle["categorical"].items():
    row[col] = st.selectbox(col, options)

if st.button("Predict rent"):
    x = pd.get_dummies(pd.DataFrame([row]), dtype=float)
    x = x.reindex(columns=bundle["columns"], fill_value=0.0)
    rent = max(model.predict(x)[0], 0)
    st.success(f"Estimated rent: {rent:,.0f} Taka / month")
    st.caption("Eta ekta estimate, exact bhara na. Dataset ar model er limitation ache.")
