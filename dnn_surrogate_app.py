import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from io import BytesIO
from dnn_surrogate_predictor import SurrogatePredictor

# =======================================================
# Modern Page Configuration
# =======================================================
st.set_page_config(
    page_title="DNN Surrogate Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =======================================================
# Custom Modern CSS Styling
# =======================================================
st.markdown("""
<style>

    /* ---------- Global Styling ---------- */
    .main {
        background-color: #f7f9fc;
    }
    div.block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---------- Beautiful Header ---------- */
    .header-box {
        padding: 25px 20px;
        background: linear-gradient(90deg, #004E92, #000428);
        color: white;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* ---------- Instruction Card ---------- */
    .card {
        background-color: white;
        padding: 20px 25px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-bottom: 20px;
        color: #333;
    }

    /* ---------- Footer ---------- */
    .footer {
        font-size: 13px;
        text-align: center;
        margin-top: 30px;
        color: #777;
    }

    /* ---------- Buttons ---------- */
    .stButton>button {
        border-radius: 10px;
        height: 3rem;
        background: #004E92;
        color: white;
        border: none;
        font-weight: 600;
        transition: 0.2s ease-in-out;
    }

    .stButton>button:hover {
        transform: scale(1.03);
        background: #003b70;
    }

</style>
""", unsafe_allow_html=True)

# =======================================================
# Header UI
# =======================================================
st.markdown("""
<div class="header-box">
    <h1>⚡DNN Surrogate for AC-OPF Reactive Power Prediction</h1>
    <h4>Fast & Accurate PV Reactive Power Estimation using Deep Learning</h4>
</div>
""", unsafe_allow_html=True)

# =======================================================
# Sidebar – Author Profile
# =======================================================
st.sidebar.markdown("""
### 👤 **Author**
**Morufdeen Atilola**  
PhD Student, Electrical Engineering  
University at Buffalo  

---

### ⚙ Model Status
""")

# Cache model
@st.cache_resource
def load_model():
    return SurrogatePredictor()

predictor = load_model()
st.sidebar.success("Model Loaded Successfully")

# =======================================================
# Input Instructions Card
# =======================================================
st.markdown("""
<div class="card">

### 📘 Input Requirements (130 Features)

Your CSV can be **with or without header**.

The features must follow this order:

- **P_load**: 33 values (bus 1–33)  
- **Q_load**: 33 values (bus 1–33)  
- **P_PV**: 32 values (bus 2–33)  
  - PV buses → actual values  
  - Non-PV buses → 0  
- **PV_mask**: 32 values (bus 2–33)  
  - PV bus → 1  
  - Non-PV bus → 0  

✔ Any number of rows (snapshots) allowed  
✔ Must be exactly **130 columns**

</div>
""", unsafe_allow_html=True)

# =======================================================
# Template Download Button
# =======================================================
sample_template = pd.DataFrame(np.zeros((1, 130)))
st.download_button(
    "📄 Download Input Template (CSV)",
    data=sample_template.to_csv(index=False).encode("utf-8"),
    file_name="surrogate_input_template.csv",
    mime="text/csv"
)

# =======================================================
# File Upload UI
# =======================================================
uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Try loading file with or without header
    try:
        df = pd.read_csv(uploaded_file)
        if df.shape[1] != 130:
            df = pd.read_csv(uploaded_file, header=None)
    except:
        df = pd.read_csv(uploaded_file, header=None)

    # Validation
    if df.shape[1] != 130:
        st.error(f"❌ File contains {df.shape[1]} columns. Expected exactly 130.")
        st.stop()

    # Show uploaded data
    st.markdown("""
    <div class="card">
        <h4>📊 Uploaded Data Preview</h4>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(df.head())

    # =============================================
    # Run Prediction Button
    # =============================================
    if st.button("⚡ Run Prediction"):
        start_time = time.time()

        try:
            outputs = predictor.predict_q_table(df)
            runtime = time.time() - start_time

            st.markdown("""
            <div class="card">
                <h3>✅ Prediction Output</h3>
            </div>
            """, unsafe_allow_html=True)

            # Single snapshot (DataFrame)
            if isinstance(outputs, pd.DataFrame):
                st.dataframe(outputs.style.format({"Predicted Q* (Mvar)": "{:.4f}"}))

                # CSV download
                csv_bytes = outputs.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "💾 Download Q* Predictions (CSV)",
                    data=csv_bytes,
                    file_name="q_predictions.csv",
                    mime="text/csv"
                )

                # Plot
                fig, ax = plt.subplots(figsize=(8,4))
                ax.bar(outputs["PV Bus #"], outputs["Predicted Q* (Mvar)"], color="#005DAA88")
                ax.set_xlabel("PV Bus #")
                ax.set_ylabel("Q* (Mvar)")
                ax.set_title("Predicted Reactive Power for Snapshot")
                ax.grid(False)
                st.pyplot(fig)

                # PNG and PDF downloads
                png_buf = BytesIO()
                pdf_buf = BytesIO()
                fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
                fig.savefig(pdf_buf, format="pdf", bbox_inches="tight")
                st.download_button(
                    "📥 Download Plot (PNG)",
                    png_buf.getvalue(),
                    file_name="q_prediction.png",
                    mime="image/png"
                )
                st.download_button(
                    "📥 Download Plot (PDF)",
                    pdf_buf.getvalue(),
                    file_name="q_prediction.pdf",
                    mime="application/pdf"
                )

            # Multiple snapshots (list of DataFrames)
            else:
                for i, snap_df in enumerate(outputs, start=1):
                    st.markdown(f"### Snapshot {i}")
                    st.dataframe(snap_df.style.format({"Predicted Q* (Mvar)": "{:.4f}"}))

                    # Export CSV for this snapshot
                    csv_bytes = snap_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        f"💾 Download Snapshot {i} (CSV)",
                        csv_bytes,
                        file_name=f"q_predictions_snapshot_{i}.csv",
                        mime="text/csv"
                    )

                    # Plot each snapshot
                    fig, ax = plt.subplots(figsize=(8,4))
                    ax.bar(snap_df["PV Bus #"], snap_df["Predicted Q* (Mvar)"], color="#005DAA88")
                    ax.set_xlabel("PV Bus #")
                    ax.set_ylabel("Q* (Mvar)")
                    ax.set_title(f"Snapshot {i} — Predicted Reactive Power")
                    ax.grid(False)
                    st.pyplot(fig)

                    # PNG/PDF downloads
                    buf_png = BytesIO()
                    buf_pdf = BytesIO()
                    fig.savefig(buf_png, format="png", dpi=300, bbox_inches="tight")
                    fig.savefig(buf_pdf, format="pdf", bbox_inches="tight")

                    st.download_button(
                        f"📥 Download Plot (PNG) — Snapshot {i}",
                        buf_png.getvalue(),
                        file_name=f"q_prediction_snapshot_{i}.png",
                        mime="image/png"
                    )
                    st.download_button(
                        f"📥 Download Plot (PDF) — Snapshot {i}",
                        buf_pdf.getvalue(),
                        file_name=f"q_prediction_snapshot_{i}.pdf",
                        mime="application/pdf"
                    )

            st.success(f"🕒 Execution Time: {runtime:.4f} seconds")

        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")

# =======================================================
# Footer
# =======================================================
st.markdown("""
<div class="footer">
Powered by the DNN Surrogate Model — University at Buffalo  
</div>
""", unsafe_allow_html=True)