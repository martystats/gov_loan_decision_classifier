import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ----------------------------
# Paths (robust)
# ----------------------------
APP_DIR = Path(__file__).resolve().parent               # .../gov_loan_amount_classifier/app
PROJECT_DIR = APP_DIR.parent                            # .../gov_loan_amount_classifier
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
DATA_DIR = PROJECT_DIR / "data"

MODEL_PATH = ARTIFACTS_DIR / "gb_tuned_model.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "feature_columns.pkl"
CATEGORIES_PATH = DATA_DIR / "categories.json"


# ----------------------------
# Load artifacts
# ----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)

    categories = {}
    if CATEGORIES_PATH.exists():
        with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
            categories = json.load(f)

    return model, feature_cols, categories


model, feature_columns, categories = load_artifacts()


# ----------------------------
# Helpers
# ----------------------------
def one_hot_col(prefix: str, value: str) -> str:
    # matches training dummy naming: e.g. country_Angola, policy_type_EDS
    return f"{prefix}_{value}"


def build_feature_row(raw_inputs: dict, feature_columns: list) -> pd.DataFrame:
    """
    raw_inputs:
      - fiscal_year: int
      - brokered_Yes: bool
      - deal_cancelled_Yes: bool
      - country: str or None
      - decision_authority: str or None
      - policy_type: str or None
      - program: str or None
      - term: str or None

    Returns a 1-row dataframe with columns EXACTLY matching feature_columns.
    """
    row = pd.DataFrame([{c: 0 for c in feature_columns}])

    # numeric
    if "fiscal_year" in row.columns:
        row.loc[0, "fiscal_year"] = int(raw_inputs.get("fiscal_year", 0))

    # binary (these must match your feature_columns)
    for bcol in ["brokered_Yes", "deal_cancelled_Yes"]:
        if bcol in row.columns:
            row.loc[0, bcol] = int(bool(raw_inputs.get(bcol, False)))

    # categorical -> set the correct dummy column to 1 IF it exists
    for cat in ["country", "decision_authority", "policy_type", "program", "term"]:
        val = raw_inputs.get(cat)
        if val is None:
            continue

        dummy_name = one_hot_col(cat, val)
        if dummy_name in row.columns:
            row.loc[0, dummy_name] = 1
        else:
            # This is IMPORTANT:
            # If training used drop_first=True, the "baseline" category has NO dummy column.
            # Example: program has only program_Working Capital.
            # If user selects Insurance and there is no program_Insurance column, we do nothing (stays 0) -> correct.
            pass

    return row


# ----------------------------
# UI
# ----------------------------
st.title("Government Loan Amount Classification System")
st.write("Predict whether a loan belongs to a **High vs Low** loan amount category (based on the trained model).")

with st.expander("Inputs (User Friendly)", expanded=True):

    st.subheader("Categorical (Dropdowns)")

    # Dropdown options come from categories.json
    def get_opts(key):
        opts = categories.get(key, [])
        # force strings
        opts = [str(x) for x in opts]
        return ["(None)"] + opts

    country_sel = st.selectbox("Country", get_opts("country"))
    decision_sel = st.selectbox("Decision Authority", get_opts("decision_authority"))
    policy_sel = st.selectbox("Policy Type", get_opts("policy_type"))
    program_sel = st.selectbox("Program", get_opts("program"))
    term_sel = st.selectbox("Term", get_opts("term"))

    st.subheader("Binary (Checkboxes)")
    brokered_yes = st.checkbox("Brokered = Yes")
    deal_cancelled_yes = st.checkbox("Deal Cancelled = Yes")

    st.subheader("Numeric (User Friendly)")

    # User-friendly fiscal year (slider)
    # If you have many years in your dataset, this is easier than +/- buttons.
    year_min, year_max = 1990, 2030
    # If categories.json contains fiscal years as strings in another key, you can adapt later.
    fiscal_year = st.slider("Fiscal Year", min_value=year_min, max_value=year_max, value=2011, step=1)


raw_inputs = {
    "fiscal_year": fiscal_year,
    "brokered_Yes": brokered_yes,
    "deal_cancelled_Yes": deal_cancelled_yes,
    "country": None if country_sel == "(None)" else country_sel,
    "decision_authority": None if decision_sel == "(None)" else decision_sel,
    "policy_type": None if policy_sel == "(None)" else policy_sel,
    "program": None if program_sel == "(None)" else program_sel,
    "term": None if term_sel == "(None)" else term_sel,
}

if st.button("Predict"):
    X = build_feature_row(raw_inputs, feature_columns)
    pred = model.predict(X)[0]

    # If your model outputs 0/1, map it:
    # Adjust if your classes are reversed.
    if pred in [0, 1]:
        label = "High" if pred == 1 else "Low"
    else:
        label = str(pred)

    st.subheader("Prediction Result")
    st.write(f"**Predicted Class:** {label}")

    with st.expander("Debug (see final input row)"):
        st.write("Loaded categories.json path:", str(CATEGORIES_PATH))
        st.write("Country options count:", len(categories.get("country", [])))
        st.dataframe(X)