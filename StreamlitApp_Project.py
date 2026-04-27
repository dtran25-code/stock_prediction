"""
Streamlit app — IEEE-CIS Fraud Detection.

Two execution modes (auto-selected):
  • LIVE  — calls the SageMaker endpoint via boto3, when AWS creds + endpoint
            name are present in Streamlit secrets.
  • LOCAL — loads `fine_tuned_pipeline.joblib` from disk and runs locally.
            Used when AWS creds are missing/expired (e.g. Streamlit Cloud
            without secrets, or local dev). Lets the app keep working
            after Learner Lab credentials expire.

Required Streamlit secrets (Streamlit Cloud → app → Settings → Secrets):
  [aws_credentials]
  AWS_ACCESS_KEY_ID     = "..."
  AWS_SECRET_ACCESS_KEY = "..."
  AWS_SESSION_TOKEN     = "..."        # required for AWS Academy temp creds
  AWS_BUCKET            = "sagemaker-us-east-1-..."
  AWS_ENDPOINT          = "fraud-detection-endpoint"

Local dev: write the same block to .streamlit/secrets.toml (gitignored).
"""

import os
import sys
import warnings
import tempfile
import posixpath
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import shap

warnings.simplefilter("ignore")

# ── Path setup so `src.Custom_Classes` is importable ────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.Custom_Classes import *  # noqa: F401,F403  unpickle hook

# ── Mode detection: LIVE (SageMaker) vs LOCAL (joblib) ─────────────────────
def _have_aws_secrets() -> bool:
    try:
        c = st.secrets["aws_credentials"]
        return bool(c.get("AWS_ACCESS_KEY_ID")
                    and c.get("AWS_SECRET_ACCESS_KEY")
                    and c.get("AWS_ENDPOINT"))
    except Exception:
        return False

LIVE_MODE = _have_aws_secrets()

# ── Model + explainer loaders (cached so the app stays fast) ───────────────
@st.cache_resource(show_spinner="Loading model…")
def load_local_pipeline():
    p = PROJECT_ROOT / "fine_tuned_pipeline.joblib"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Either populate AWS secrets to run in LIVE mode, "
            "or place the .joblib next to this app for LOCAL mode."
        )
    return joblib.load(p)

@st.cache_resource(show_spinner="Loading SHAP explainer…")
def load_local_explainer():
    p = PROJECT_ROOT / "explainer_fraud.shap"
    return joblib.load(p) if p.exists() else None

@st.cache_resource(show_spinner="Connecting to SageMaker…")
def get_predictor():
    """Return (predictor, sm_session) for LIVE mode."""
    import boto3, sagemaker
    from sagemaker.predictor import Predictor
    from sagemaker.serializers import JSONSerializer
    from sagemaker.deserializers import JSONDeserializer

    creds = st.secrets["aws_credentials"]
    session = boto3.Session(
        aws_access_key_id     = creds["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key = creds["AWS_SECRET_ACCESS_KEY"],
        aws_session_token     = creds.get("AWS_SESSION_TOKEN") or None,
        region_name           = creds.get("AWS_REGION", "us-east-1"),
    )
    sm_session = sagemaker.Session(boto_session=session)
    predictor = Predictor(
        endpoint_name      = creds["AWS_ENDPOINT"],
        sagemaker_session  = sm_session,
        serializer         = JSONSerializer(),
        deserializer       = JSONDeserializer(),
    )
    return predictor

# ── Template row: the merged-dataset's first row, used to fill columns the
#    user doesn't expose in the form. We ship it next to the app so we don't
#    re-merge the CSVs at request time. ─────────────────────────────────────
@st.cache_data(show_spinner="Loading template row…")
def load_template_row():
    """Return one merged transaction+identity row, dropped of `isFraud`."""
    p = PROJECT_ROOT / "_template_row.csv"
    if p.exists():
        return pd.read_csv(p)
    # Fallback: merge on the fly from the raw CSVs (slower).
    tt = pd.read_csv(PROJECT_ROOT / "train_transaction.csv", nrows=1)
    ti = pd.read_csv(PROJECT_ROOT / "train_identity.csv")
    row = tt.merge(ti, on="TransactionID", how="left").drop(columns=["isFraud"])
    return row.head(1)

# ── User input config — picked for fraud relevance & user-friendliness ─────
USER_INPUTS = [
    {"name": "TransactionAmt", "type": "number",  "min": 1.0,    "max": 5000.0, "default": 75.0,  "step": 0.50},
    {"name": "ProductCD",      "type": "select",  "options": ["W","C","R","H","S"],                          "default": "W"},
    {"name": "card4",          "type": "select",  "options": ["visa","mastercard","american express","discover"], "default": "visa"},
    {"name": "card6",          "type": "select",  "options": ["debit","credit","debit or credit","charge card"],  "default": "credit"},
    {"name": "addr1",          "type": "number",  "min": 100.0,  "max": 540.0,  "default": 299.0, "step": 1.0},
    {"name": "addr2",          "type": "number",  "min": 80.0,   "max": 100.0,  "default": 87.0,  "step": 1.0},
    {"name": "dist1",          "type": "number",  "min": 0.0,    "max": 5000.0, "default": 8.0,   "step": 1.0},
    {"name": "P_emaildomain",  "type": "select",  "options": ["gmail.com","yahoo.com","hotmail.com","aol.com","comcast.net","icloud.com","anonymous.com"], "default": "gmail.com"},
    {"name": "C1",             "type": "number",  "min": 0.0,    "max": 800.0,  "default": 1.0,   "step": 1.0},
    {"name": "C2",             "type": "number",  "min": 0.0,    "max": 800.0,  "default": 1.0,   "step": 1.0},
    {"name": "D1",             "type": "number",  "min": 0.0,    "max": 700.0,  "default": 14.0,  "step": 1.0},
    {"name": "M1",             "type": "select",  "options": ["T","F"],                                       "default": "T"},
]

# ── Prediction helpers ──────────────────────────────────────────────────────
def predict_live(input_df: pd.DataFrame):
    predictor = get_predictor()
    payload   = input_df.to_json(orient="records")
    raw       = predictor.predict(payload)
    # raw is a list (output_fn returns json.dumps(list))
    val = int(np.asarray(raw).ravel()[0])
    return val

def predict_local(input_df: pd.DataFrame):
    pipe = load_local_pipeline()
    val  = int(pipe.predict(input_df)[0])
    return val

def predict(input_df: pd.DataFrame):
    """Try LIVE first; on failure, fall back to LOCAL."""
    if LIVE_MODE:
        try:
            return predict_live(input_df), "LIVE  (SageMaker endpoint)"
        except Exception as e:
            st.warning(f"SageMaker call failed ({type(e).__name__}). Falling back to local model.")
    return predict_local(input_df), "LOCAL (joblib pipeline)"

def show_shap_explanation(input_df: pd.DataFrame):
    pipe      = load_local_pipeline()
    explainer = load_local_explainer()
    if explainer is None:
        st.info("SHAP explainer file not found locally — explanations unavailable.")
        return
    pre   = pipe[:-3]                # all transforms up to and including SelectKBest
    X_pre = pre.transform(input_df)
    try:
        feat_names = pre.get_feature_names_out()
    except Exception:
        feat_names = np.array([f"f{i}" for i in range(X_pre.shape[1])])
    X_pre_df = pd.DataFrame(X_pre, columns=feat_names)
    sv       = explainer(X_pre_df)
    sv_fraud = sv[..., 1] if len(sv.shape) == 3 else sv

    st.subheader("Decision transparency — SHAP")
    fig = plt.figure(figsize=(10, 5))
    shap.plots.waterfall(sv_fraud[0], max_display=12, show=False)
    plt.tight_layout()
    st.pyplot(fig)

    top_feat = pd.Series(sv_fraud[0].values, index=sv_fraud[0].feature_names).abs().idxmax()
    st.info(f"Most influential factor in this prediction: **{top_feat}**")

# ── UI ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("IEEE-CIS Fraud Detection")

mode_label = "LIVE  ✓ SageMaker endpoint" if LIVE_MODE else "LOCAL  (no AWS creds — running joblib)"
st.caption(f"Inference mode: **{mode_label}**")

with st.form("pred_form"):
    st.subheader("Transaction details")
    cols = st.columns(2)
    user_vals = {}
    for i, inp in enumerate(USER_INPUTS):
        with cols[i % 2]:
            label = inp["name"].replace("_", " ").title()
            if inp["type"] == "number":
                user_vals[inp["name"]] = st.number_input(
                    label,
                    min_value=inp["min"], max_value=inp["max"],
                    value=inp["default"], step=inp["step"],
                )
            else:
                user_vals[inp["name"]] = st.selectbox(
                    label, options=inp["options"],
                    index=inp["options"].index(inp["default"]),
                )
    submitted = st.form_submit_button("Run Prediction")

if submitted:
    template = load_template_row().copy()
    for col, val in user_vals.items():
        if col in template.columns:
            template.at[template.index[0], col] = val

    pred_val, mode_used = predict(template)
    label = "Fraud" if pred_val == 1 else "Legitimate"

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Prediction", label)
        st.caption(f"Inference path: {mode_used}")
    with c2:
        st.subheader("Inputs sent to model")
        st.json(user_vals)

    show_shap_explanation(template)
