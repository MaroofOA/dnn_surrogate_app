import streamlit as st
import pandas as pd
import numpy as np
import time
from dnn_surrogate_predictor import SurrogatePredictor

# -----------------------------
# Modern Page Config
# -----------------------------
st.set_page_config(
    page_title="DNN Surrogate Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS (Modern Look)
# -----------------------------
st.markdown("""
    <style>
        .main { background-color: #F8F9FA; }
        .stButton>button {
            background-color: #005DAA;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #00457D;
        }
        .title-banner {
            background: linear-gradient(90deg, #005DAA, #00994C);
            padding: 18px;
            border-radius: 10px;
            color: white;
            text-align: center;
            font-size: 26px;
            font-weight: bold;
        }
        .card {
            background-color: white;
            padding: 22px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Title Section
# -----------------------------
st.markdown('<div class="title-banner">DNN Surrogate Reactive Power Predictor</div>',
            unsafe_allow_html=True)

st.markdown("### Upload your input data to obtain predicted inverter reactive power setpoints.")

# -----------------------------
# Template Download
# -----------------------------
sample_template = pd.DataFrame(np.zeros((1,130)))
csv = sample_template.to_csv(index=False).encode("utf-8")

with st.expander("📄 Download Input Template"):
    st.download_button(
        label="Download CSV Template (130 columns)",
        data=csv,
        file_name="surrogate_input_template.csv",
        mime="text/csv"
    )

# -----------------------------
# File Upload Section
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📤 Upload CSV file containing input features (each row = snapshot)",
    type=["csv"]
)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return SurrogatePredictor()

predictor = load_model()

# -----------------------------
# Run Prediction
# -----------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"File uploaded successfully with shape {df.shape}")

    if st.button("⚡ Run Prediction"):
        start = time.time()

        try:
            outputs = predictor.predict_q_table(df)
            exec_time = time.time() - start

            st.markdown("### ✅ Prediction Results")
            
            # Single snapshot
            if isinstance(outputs, pd.DataFrame):
                st.dataframe(outputs.style.format({"Predicted Q* (Mvar)": "{:.4f}"}))
            
            # Multiple snapshots
            else:
                for i, table in enumerate(outputs):
                    st.markdown(f"**Snapshot {i+1}**")
                    st.dataframe(table.style.format({"Predicted Q* (Mvar)": "{:.4f}"}))
                    st.markdown("---")

            # -----------------------------
            # Execution time display
            # -----------------------------
            st.info(f"🕒 Prediction completed in **{exec_time:.4f} seconds**.")

        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")