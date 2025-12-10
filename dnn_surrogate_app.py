import streamlit as st
import pandas as pd
import numpy as np
import time
from dnn_surrogate_predictor import SurrogatePredictor

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(
    page_title="DNN Surrogate Predictor – Atilola",
    layout="centered"
)

# -----------------------------
# Title + Description
# -----------------------------
st.markdown("""
# ⚡ DNN Surrogate Reactive Power Predictor  
### Predict PV Inverter Reactive Power Setpoints using a Trained Deep Neural Network  
---
""")

# -----------------------------
# Input Instructions
# -----------------------------
st.markdown("""
### 📘 **Input Data Requirements**

Your uploaded CSV must contain **exactly 130 features per snapshot**, ordered as:

#### **1. Load Features**
- **P_load**: 33 values → *buses 1–33*  
- **Q_load**: 33 values → *buses 1–33*

#### **2. PV Injection Features**
- **P_PV**: 32 values → *buses 2–33*  
  - PV-connected buses: actual P_PV values  
  - Non-PV buses: **0**

#### **3. PV Mask**
- **PV_mask**: 32 values → *buses 2–33*  
  - PV-connected bus → **1**  
  - Non-PV bus → **0**

#### ✔ CSV can be uploaded **with or without headers**  
The app automatically detects formats.

---

""")

# -----------------------------
# Template Download
# -----------------------------
sample_template = pd.DataFrame(np.zeros((1,130)))
st.download_button(
    label="📄 Download Input Template (CSV)",
    data=sample_template.to_csv(index=False).encode("utf-8"),
    file_name="surrogate_input_template.csv",
    mime="text/csv"
)

# -----------------------------
# Sidebar – Author Info
# -----------------------------
st.sidebar.markdown("""
### 👤 **Author**
**Morufdeen Atilola**  
PhD Student, Electrical Engineering  
University at Buffalo

---
""")

# -----------------------------
# Load Model (Cached)
# -----------------------------
@st.cache_resource
def load_model():
    return SurrogatePredictor()

predictor = load_model()

# -----------------------------
# File Upload Area
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload CSV containing the 130 input features", type=["csv"])

if uploaded_file is not None:
    # Try reading normally
    try:
        df = pd.read_csv(uploaded_file)
        # If headerless, fix it
        if df.shape[1] != 130:
            df = pd.read_csv(uploaded_file, header=None)
    except:
        df = pd.read_csv(uploaded_file, header=None)

    # Validate correct dimension
    if df.shape[1] != 130:
        st.error(f"❌ Input data must contain exactly **130 columns**, but file contains **{df.shape[1]}**.")
        st.stop()

    st.success(f"File uploaded successfully: {df.shape[0]} snapshot(s) × {df.shape[1]} features")

    # Display input
    st.markdown("### 🔍 Uploaded Input Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Run Prediction
    # -----------------------------
    if st.button("⚡ Run Prediction"):
        start = time.time()

        try:
            results = predictor.predict_q_table(df)
            exec_time = time.time() - start

            st.markdown("## ✅ Prediction Output")

            # Single snapshot
            if isinstance(results, pd.DataFrame):
                st.dataframe(
                    results.style.format({"Predicted Q* (Mvar)": "{:.4f}"})
                )

            # Multiple snapshots
            else:
                for i, r in enumerate(results):
                    st.markdown(f"### Snapshot {i+1}")
                    st.dataframe(
                        r.style.format({"Predicted Q* (Mvar)": "{:.4f}"})
                    )
                    st.markdown("---")

            st.info(f"🕒 Prediction completed in **{exec_time:.4f} seconds**.")

        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")