# ---------------------------
# Fixed SurrogatePredictor class (copy-paste this whole cell)
# ---------------------------
import os
import glob
import joblib
import torch
import numpy as np
import pandas as pd
from typing import Union

# display for Jupyter safe-printing
try:
    from IPython.display import display
except Exception:
    display = print

import torch.nn as nn
class SurrogateDNN(nn.Module):
    def __init__(self, input_dim=130, hidden=[256,128,64], output_dim=32, dropout=0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# constants (must match your training config)
S_BASE = 100.0
INPUT_DIM = 130
OUTPUT_DIM = 32
IDX_PV_P_START = 66
IDX_PV_P_END = 98
IDX_PV_MASK_START = 98
IDX_PV_MASK_END = 130

class SurrogatePredictor:
    def __init__(self, artifact_path: str = None, model_hidden=[256,128,64], dropout: float = 0.0):
        """
        Load latest model and scalers from model_artifacts (auto-detected if not provided).
        """
        # device as instance attribute (important)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # locate artifacts folder
        if artifact_path is None:
            cand = os.path.join(os.getcwd(), "model_artifacts")
            if os.path.isdir(cand):
                artifact_path = cand
            else:
                raise FileNotFoundError(
                    "model_artifacts folder not found in the current working directory. "
                    "Either move this notebook/script into the folder containing model_artifacts "
                    "or pass artifact_path explicitly to SurrogatePredictor(...)."
                )
        self.artifact_path = artifact_path
        self.model_hidden = model_hidden
        self.dropout = dropout

        # check scaler files
        x_scaler_path = os.path.join(self.artifact_path, "x_scaler.joblib")
        y_scaler_path = os.path.join(self.artifact_path, "y_scaler.joblib")
        if not os.path.exists(x_scaler_path) or not os.path.exists(y_scaler_path):
            raise FileNotFoundError(f"Missing x_scaler.joblib or y_scaler.joblib in {self.artifact_path}")

        # load scalers
        self.x_scaler = joblib.load(x_scaler_path)
        self.y_scaler = joblib.load(y_scaler_path)

        # find latest model (.pth)
        model_files = sorted(glob.glob(os.path.join(self.artifact_path, "*.pth")), key=os.path.getmtime, reverse=True)
        if not model_files:
            raise FileNotFoundError(f"No .pth model file found in {self.artifact_path}")
        latest_model = model_files[0]

        # build model and load weights (make sure architecture matches the saved checkpoint)
        self.model = SurrogateDNN(input_dim=INPUT_DIM, hidden=self.model_hidden, output_dim=OUTPUT_DIM, dropout=self.dropout).to(self.device)
        self.model.load_state_dict(torch.load(latest_model, map_location=self.device))
        self.model.eval()

        print(f"Loaded model: {os.path.basename(latest_model)}")
        print(f"Model and scalers loaded from: {os.path.abspath(self.artifact_path)}")
        print(f"Using device: {self.device}\n")

    def predict_q_table(self, data: Union[list, np.ndarray, pd.DataFrame]):
        """
        Predict Q* for single snapshot (list / 1D np.array) or multiple snapshots (2D np.array or DataFrame).
        - Converts inputs to per-unit BEFORE scaling (critical).
        - Returns:
            - single DataFrame with columns ['PV Bus #', 'Predicted Q* (Mvar)'] if input is a single snapshot
            - list of DataFrames (and prints each) if input contains multiple snapshots
        """

        # normalize input to pandas DataFrame for consistent indexing
        if isinstance(data, (list, np.ndarray)) and (not isinstance(data, pd.DataFrame)):
            arr = np.array(data, dtype=float)
            if arr.ndim == 1:
                df = pd.DataFrame([arr])
            elif arr.ndim == 2:
                df = pd.DataFrame(arr)
            else:
                raise ValueError("Numpy input must be 1D or 2D array.")
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError("Input must be list, numpy array, or pandas DataFrame.")

        # sanity check feature dimension
        if df.shape[1] != INPUT_DIM:
            raise ValueError(f"Each row must have {INPUT_DIM} features. Received rows with {df.shape[1]} columns.")

        # --- Convert to per-unit (this fixes the magnitude mismatch) ---
        X_pu = df.values.astype(float) / S_BASE

        # --- Scale (x_scaler was fit on per-unit training data) ---
        X_scaled = self.x_scaler.transform(X_pu).astype(np.float32)

        # --- Model inference (in batches if necessary) ---
        xb = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            y_scaled = self.model(xb).cpu().numpy()

        # --- Inverse transform predictions and convert to Mvar ---
        y_pu = self.y_scaler.inverse_transform(y_scaled)
        y_Mvar = y_pu * S_BASE  # shape: (n_samples, OUTPUT_DIM)

        # --- Extract PV mask from original data (in physical units) ---
        pv_mask = df.values[:, IDX_PV_MASK_START:IDX_PV_MASK_END]   # shape: (n_samples, 32)

        # --- Single snapshot handling ---
        n_samples = y_Mvar.shape[0]
        if n_samples == 1:
            active_idx = np.where(pv_mask[0] == 1)[0]
            if active_idx.size == 0:
                print("⚠️ No active PV buses found for this snapshot.")
                return pd.DataFrame(columns=["PV Bus #", "Predicted Q* (Mvar)"])
            df_out = pd.DataFrame({
                "PV Bus #": active_idx + 1,
                "Predicted Q* (Mvar)": np.round(y_Mvar[0, active_idx], 4)
            })
            return df_out

        # --- Multiple snapshots: print each snapshot table and return list of DataFrames ---
        outputs = []
        for s in range(n_samples):
            active_idx = np.where(pv_mask[s] == 1)[0]
            if active_idx.size == 0:
                print(f"⚠️ No active PV buses found for Snapshot {s+1}.")
                outputs.append(pd.DataFrame(columns=["PV Bus #", "Predicted Q* (Mvar)"]))
                continue
            df_snap = pd.DataFrame({
                "PV Bus #": active_idx + 1,
                "Predicted Q* (Mvar)": np.round(y_Mvar[s, active_idx], 4)
            })
            print(f"\n===== Snapshot {s+1} =====")
            display(df_snap)
            outputs.append(df_snap)
        return outputs