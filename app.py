import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

from dnn_surrogate_predictor import SurrogatePredictor

# ---------------------------------------------------------
# Load model (cached so Streamlit doesn't reload each run)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return SurrogatePredictor()   # auto-detects model_artifacts

model = load_model()

# ---------------------------------------------------------
# Streamlit UI Setup
# ---------------------------------------------------------
st.title("🔌 DNN Surrogate for AC-OPF Reactive Power Prediction")
st.write("""
Upload a CSV containing **input feature vectors**  
(\(130\) features per row, matching your trained surrogate).  
The surrogate will compute the predicted reactive power injections \(Q^*\) for all active PV buses.
""")

uploaded_file = st.file_uploader("Upload input CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Input Data")
    st.dataframe(df)

    # ---------------------------------------
    # Run Prediction
    # ---------------------------------------
    if st.button("⚡ Predict Reactive Power"):
        outputs = model.predict_q_table(df)

        st.subheader("📊 Prediction Results")

        # Single snapshot → DataFrame
        if isinstance(outputs, pd.DataFrame):
            st.dataframe(outputs)

            # Export CSV
            csv_bytes = outputs.to_csv(index=False).encode("utf-8")
            st.download_button(
                "💾 Download Q* Predictions (CSV)",
                data=csv_bytes,
                file_name="q_predictions.csv",
                mime="text/csv"
            )

            # Plot
            fig, ax = plt.subplots(figsize=(7,4))
            ax.bar(outputs["PV Bus #"], outputs["Predicted Q* (Mvar)"],
                   color="#005DAA88")  # blue, 60% transparent
            ax.set_xlabel("PV Bus #")
            ax.set_ylabel("Q* (Mvar)")
            ax.set_title("Predicted Reactive Power for Snapshot")
            ax.grid(False)

            st.pyplot(fig)

            # PNG Download
            png_buf = BytesIO()
            fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
            st.download_button(
                "📥 Download Plot (PNG)",
                png_buf.getvalue(),
                file_name="q_prediction.png",
                mime="image/png"
            )

            # PDF download
            pdf_buf = BytesIO()
            fig.savefig(pdf_buf, format="pdf", bbox_inches="tight")
            st.download_button(
                "📥 Download Plot (PDF)",
                pdf_buf.getvalue(),
                file_name="q_prediction.pdf",
                mime="application/pdf"
            )

        # Multiple snapshots → list of DataFrames
        else:
            for i, snap_df in enumerate(outputs, start=1):
                st.write(f"### Snapshot {i}")
                st.dataframe(snap_df)

                # Export CSV
                csv_bytes = snap_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    f"💾 Download Snapshot {i} (CSV)",
                    csv_bytes,
                    file_name=f"q_predictions_snapshot_{i}.csv",
                    mime="text/csv"
                )

                # Plot each snapshot
                fig, ax = plt.subplots(figsize=(7,4))
                ax.bar(snap_df["PV Bus #"], snap_df["Predicted Q* (Mvar)"],
                       color="#005DAA88")
                ax.set_xlabel("PV Bus #")
                ax.set_ylabel("Q* (Mvar)")
                ax.set_title(f"Snapshot {i} — Predicted Reactive Power")
                ax.grid(False)
                st.pyplot(fig)

                # PNG/PDF download
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
