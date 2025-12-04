import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tempfile
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# -----------------------------------------------------------
# DARK THEME CSS
# -----------------------------------------------------------

def load_css():
    custom_css = """    
    .stApp {
        background-color: #0e0e0e !important;
        color: #f1f1f1 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    h1, h2, h3, h4, h5, h6, label {
        color: #ffffff !important;
    }
    .stButton > button {
        background-color: #444 !important;
        color: white !important;
        padding: 8px 18px !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background-color: #666 !important;
    }
    input, select, textarea {
        background-color: #222 !important;
        color: white !important;
        border-radius: 6px !important;
        border: 1px solid #444 !important;
    }
    """
    st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

load_css()

# -----------------------------------------------------------
# CLEANING FUNCTION
# -----------------------------------------------------------

def clean_dataframe(df):
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].replace(" ", "0.0")
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)

    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)

    return df

# -----------------------------------------------------------
# LOAD SHARED ENCODERS
# -----------------------------------------------------------

try:
    with open("encoders.pk1", "rb") as f:
        encoders = pickle.load(f)
except Exception as e:
    st.error("❌ Failed to load encoders.pk1 from repo.")
    st.stop()

# -----------------------------------------------------------
# LOAD DEFAULT ML MODEL
# -----------------------------------------------------------

ml_model = None
ml_features = None

try:
    with open("customer_churn_model.pk1", "rb") as f:
        ml_data = pickle.load(f)
    ml_model = ml_data["model"]
    ml_features = ml_data["features_name"]
except Exception as e:
    st.warning("⚠ Default ML model not found or invalid.")

# -----------------------------------------------------------
# LOAD DEFAULT DL MODEL
# -----------------------------------------------------------

dl_model = None
dl_features = None

try:
    dl_model = load_model("dl_churn_model.h5")
    with open("dl_features.pk1", "rb") as f:
        dl_features = pickle.load(f)
except Exception as e:
    st.warning("⚠ Default DL model or dl_features.pk1 not found.")

# -----------------------------------------------------------
# SIDEBAR MODEL MODE & UPLOADERS
# -----------------------------------------------------------

st.sidebar.title("⚙️ Settings")

mode = st.sidebar.radio(
    "Choose Model Mode",
    [
        "Default ML model",
        "Default DL model",
        "Upload ML (.pk1)",
        "Upload DL (.h5)"
    ]
)

uploaded_ml_file = None
uploaded_dl_file = None

if mode == "Upload ML (.pk1)":
    uploaded_ml_file = st.sidebar.file_uploader("Upload ML Model (.pk1)", type=["pk1", "pkl"])

if mode == "Upload DL (.h5)":
    uploaded_dl_file = st.sidebar.file_uploader("Upload DL Model (.h5)", type=["h5"])

# -----------------------------------------------------------
# DETERMINE ACTIVE MODEL
# -----------------------------------------------------------

active_model_type = None  # "ML" or "DL"
active_model = None
active_ml_features = None
active_dl_features = dl_features  # default

# ML selection
if mode == "Default ML model":
    if ml_model is not None:
        active_model_type = "ML"
        active_model = ml_model
        active_ml_features = ml_features
    else:
        st.error("❌ Default ML model not available.")

elif mode == "Upload ML (.pk1)":
    if uploaded_ml_file is not None:
        try:
            ml_upload = pickle.load(uploaded_ml_file)
            if isinstance(ml_upload, dict) and "model" in ml_upload and "features_name" in ml_upload:
                active_model = ml_upload["model"]
                active_ml_features = ml_upload["features_name"]
            else:
                # Assume it's a bare estimator, reuse ml_features
                active_model = ml_upload
                active_ml_features = ml_features
            active_model_type = "ML"
            st.sidebar.success("✅ Uploaded ML model in use.")
        except Exception as e:
            st.sidebar.error("❌ Failed to load uploaded ML model. Using default ML if available.")
            if ml_model is not None:
                active_model_type = "ML"
                active_model = ml_model
                active_ml_features = ml_features
    else:
        if ml_model is not None:
            st.sidebar.info("ℹ No uploaded ML model. Using default ML model.")
            active_model_type = "ML"
            active_model = ml_model
            active_ml_features = ml_features

# DL selection
elif mode == "Default DL model":
    if dl_model is not None and dl_features is not None:
        active_model_type = "DL"
        active_model = dl_model
        active_dl_features = dl_features
    else:
        st.error("❌ Default DL model not available.")

elif mode == "Upload DL (.h5)":
    if uploaded_dl_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
                tmp.write(uploaded_dl_file.read())
                tmp_path = tmp.name
            active_model = load_model(tmp_path)
            active_model_type = "DL"
            active_dl_features = dl_features  # same schema as default DL
            st.sidebar.success("✅ Uploaded DL model in use.")
        except Exception as e:
            st.sidebar.error("❌ Failed to load uploaded DL model. Using default DL if available.")
            if dl_model is not None:
                active_model_type = "DL"
                active_model = dl_model
                active_dl_features = dl_features
    else:
        if dl_model is not None:
            st.sidebar.info("ℹ No uploaded DL model. Using default DL model.")
            active_model_type = "DL"
            active_model = dl_model
            active_dl_features = dl_features

if active_model_type is None or active_model is None:
    st.warning("⚠ No active model selected or available. Check your models.")
    st.stop()

# -----------------------------------------------------------
# PREDICTION HELPERS
# -----------------------------------------------------------

def predict_with_model(df_encoded: pd.DataFrame):
    """df_encoded = cleaned, encoded, but not yet column-aligned."""
    if active_model_type == "ML":
        # Ensure all ML features exist
        for col in active_ml_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_ml = df_encoded[active_ml_features]
        pred = active_model.predict(df_ml)[0]
        prob = active_model.predict_proba(df_ml)[0][1]
        return int(pred), float(prob)
    else:  # DL
        # Ensure all DL features exist
        for col in active_dl_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_dl = df_encoded[active_dl_features]
        preds = active_model.predict(df_dl)
        prob = float(preds[0][0])
        pred_class = 1 if prob >= 0.5 else 0
        return int(pred_class), prob

# -----------------------------------------------------------
# 🔮 SINGLE PREDICTION
# -----------------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    ["🔮 Single Prediction", "📄 Batch Prediction (CSV)", "ℹ About"]
)

if page == "🔮 Single Prediction":

    st.header("🔮 Predict Customer Churn")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", 0, 72)

    with col2:
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    with col3:
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        StreamingTV = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
        StreamingMovies = st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0)
    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0)

    if st.button("Predict"):
        row = {
            'gender': gender,
            'SeniorCitizen': SeniorCitizen,
            'Partner': Partner,
            'Dependents': Dependents,
            'tenure': tenure,
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
            'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges
        }

        df = pd.DataFrame([row])
        df = clean_dataframe(df)

        # Encode categorical using shared encoders
        for col, encoder in encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col].astype(str))

        pred, prob = predict_with_model(df)

        st.success(f"Prediction: {'Churn' if pred == 1 else 'No Churn'}")
        st.info(f"Churn Probability: {prob:.4f}")

# -----------------------------------------------------------
# 📄 BATCH PREDICTION
# -----------------------------------------------------------

elif page == "📄 Batch Prediction (CSV)":

    st.header("📄 Batch Churn Prediction")

    uploaded_csv = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
        st.write("Preview:", df.head())

        if "customerID" in df.columns:
            df = df.drop("customerID", axis=1)

        if st.button("Run Batch Prediction"):

            df_clean = clean_dataframe(df.copy())

            for col, encoder in encoders.items():
                if col in df_clean.columns:
                    df_clean[col] = encoder.transform(df_clean[col].astype(str))

            preds = []
            probs = []

            for i in range(len(df_clean)):
                row_df = df_clean.iloc[i:i+1].copy()
                pred, prob = predict_with_model(row_df)
                preds.append(pred)
                probs.append(prob)

            df["Churn_Pred"] = preds
            df["Churn_Prob"] = probs

            st.success("Batch prediction completed!")
            st.dataframe(df)

            st.download_button(
                "Download Results",
                df.to_csv(index=False).encode("utf-8"),
                "churn_predictions.csv",
                "text/csv"
            )

# -----------------------------------------------------------
# ℹ ABOUT
# -----------------------------------------------------------

else:
    st.header("ℹ About This App")
    st.write("""
    This app supports both **Machine Learning** and **Deep Learning** models for telecom customer churn prediction.

    **Modes:**
    - Default ML model (RandomForest)
    - Default DL model (Keras .h5)
    - Upload your own ML model (.pk1)
    - Upload your own DL model (.h5)

    Both ML and DL models use:
    - The same encoders (`encoders.pk1`)
    - The same feature schema (ML `features_name`, DL `dl_features.pk1`)

    Pages:
    - 🔮 Single Prediction: Enter one customer's details
    - 📄 Batch Prediction: Upload a CSV of many customers
    """)

