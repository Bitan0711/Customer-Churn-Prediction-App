# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import BytesIO

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("Customer Churn Predictor")
st.write("Enter customer details or upload model+encoders (optional).")

# ---------- helpers ----------
@st.cache_resource
def load_model_from_file(file_bytes):
    try:
        data = pickle.load(file_bytes)
        return data
    except Exception as e:
        st.error(f"Failed to load pickle: {e}")
        return None

def safe_load_local_model():
    """Try to load default files from working directory if present."""
    try:
        with open("customer_churn_model.pk1", "rb") as f:
            model_data = pickle.load(f)
        with open("encoders.pk1", "rb") as f:
            encoders = pickle.load(f)
        return model_data, encoders
    except FileNotFoundError:
        return None, None
    except Exception as e:
        st.warning(f"Error loading local files: {e}")
        return None, None

def prepare_input_df(raw):
    """Return DataFrame aligned with model features (no reordering required if features list is used)."""
    return pd.DataFrame([raw])

# ---------- load model & encoders (either local or uploaded) ----------
model_data = None
encoders = None

# Option A: let user upload model and encoders
uploaded_model_file = st.file_uploader("Upload `customer_churn_model.pk1` (optional)", type=["pkl","pk1","pickle"])
uploaded_encoders_file = st.file_uploader("Upload `encoders.pk1` (optional)", type=["pkl","pk1","pickle"])

if uploaded_model_file and uploaded_encoders_file:
    model_data = load_model_from_file(uploaded_model_file)
    encoders = load_model_from_file(uploaded_encoders_file)
else:
    # Option B: try to load bundled files from the app folder
    mdl, enc = safe_load_local_model()
    if mdl is not None and enc is not None:
        model_data, encoders = mdl, enc

if model_data is None:
    st.info("No model loaded. You can still try the UI in 'Dry-run' mode (prediction will be disabled).")
else:
    loaded_model = model_data["model"]
    feature_names = model_data.get("features_name", None)

# ---------- Input form for single prediction ----------
st.header("Single customer prediction")

with st.form("single_form"):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("gender", options=["Female","Male"])
        SeniorCitizen = st.selectbox("SeniorCitizen", options=[0,1], index=0)
        Partner = st.selectbox("Partner", options=["Yes","No"])
        Dependents = st.selectbox("Dependents", options=["Yes","No"])
        tenure = st.number_input("tenure (0–72 months)", min_value=0, max_value=120, value=12, step=1)
        PhoneService = st.selectbox("PhoneService", options=["Yes","No"])
        MultipleLines = st.selectbox("MultipleLines", options=["No phone service","No","Yes"])
    with col2:
        InternetService = st.selectbox("InternetService", options=["DSL","Fiber optic","No"])
        OnlineSecurity = st.selectbox("OnlineSecurity", options=["Yes","No","No internet service"])
        OnlineBackup = st.selectbox("OnlineBackup", options=["Yes","No","No internet service"])
        DeviceProtection = st.selectbox("DeviceProtection", options=["Yes","No","No internet service"])
        TechSupport = st.selectbox("TechSupport", options=["Yes","No","No internet service"])
        StreamingTV = st.selectbox("StreamingTV", options=["Yes","No","No internet service"])
        StreamingMovies = st.selectbox("StreamingMovies", options=["Yes","No","No internet service"])

    col3, col4 = st.columns(2)
    with col3:
        Contract = st.selectbox("Contract", options=["Month-to-month","One year","Two year"])
        PaperlessBilling = st.selectbox("PaperlessBilling", options=["Yes","No"])
        PaymentMethod = st.selectbox("PaymentMethod", options=[
            "Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"
        ])
    with col4:
        MonthlyCharges = st.number_input("MonthlyCharges (e.g., 29.85)", min_value=0.0, value=50.0, format="%.2f")
        TotalCharges = st.number_input("TotalCharges", min_value=0.0, value=MonthlyCharges*tenure, format="%.2f")

    submit_btn = st.form_submit_button("Predict")

if submit_btn:
    raw = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': int(tenure),
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': float(MonthlyCharges),
        'TotalCharges': float(TotalCharges)
    }

    input_df = prepare_input_df(raw)

    if encoders is None or model_data is None:
        st.error("Model or encoders not loaded. Upload them or place in app folder to get real predictions.")
    else:
        # apply encoders
        for column, encoder in encoders.items():
            if column in input_df.columns:
                try:
                    # encoder expects an array-like of strings; ensure dtype matches training
                    input_df[column] = encoder.transform(input_df[column].astype(str))
                except Exception as e:
                    st.error(f"Failed to encode column {column}: {e}")
                    st.stop()

        # reorder features if model saved feature list
        if feature_names:
            input_df = input_df[feature_names]

        # predict
        pred = loaded_model.predict(input_df)[0]
        pred_prob = loaded_model.predict_proba(input_df)[0]

        st.subheader("Result")
        st.write("Prediction:", "Churn" if pred == 1 else "No Churn")
        st.write("Probability (No Churn, Churn):", np.round(pred_prob, 4))

# ---------- Batch predictions via CSV ----------
st.header("Batch predictions (CSV)")
st.write("Upload a CSV with columns matching training features (excluding 'Churn'). If model/encoders are not loaded, batch predicts will be disabled.")

uploaded_csv = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
if uploaded_csv is not None:
    df_in = pd.read_csv(uploaded_csv)
    st.write("Preview:", df_in.head())
    if st.button("Run batch prediction"):
        if encoders is None or model_data is None:
            st.error("Model/encoders missing. Upload them to do batch predictions.")
        else:
            df_pred = df_in.copy()
            # Encode categorical columns using saved encoders
            for column, encoder in encoders.items():
                if column in df_pred.columns:
                    df_pred[column] = encoder.transform(df_pred[column].astype(str))
            if feature_names:
                df_pred = df_pred[feature_names]
            preds = loaded_model.predict(df_pred)
            probs = loaded_model.predict_proba(df_pred)[:,1]
            df_in["Churn_Pred"] = preds
            df_in["Churn_Prob"] = probs
            st.success("Done")
            st.dataframe(df_in)
            # allow download
            csv_out = df_in.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", csv_out, "predictions.csv", "text/csv")

st.markdown("---")
st.markdown("**Notes:** Put `customer_churn_model.pk1` and `encoders.pk1` in the same folder as `app.py` before launching, or upload them via the UI.")
