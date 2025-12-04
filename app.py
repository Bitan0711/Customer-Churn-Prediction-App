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
# CLEANING FUNCTION (same logic as training)
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
# LOAD DEFAULT DL MODEL + ENCODERS + FEATURE LIST
# -----------------------------------------------------------

st.sidebar.title("⚙️ Settings")

# Optionally upload a different DL model (.h5)
uploaded_model = st.sidebar.file_uploader("Upload Alternate DL Model (.h5)", type=["h5"])

# Load encoders & feature list from repo (fixed)
try:
    with open("encoders.pk1", "rb") as f:
        encoders = pickle.load(f)
    with open("dl_features.pk1", "rb") as f:
        dl_features = pickle.load(f)
except Exception as e:
    st.error("❌ Failed to load encoders or feature list from repo.")
    st.stop()

# Load default DL model
def load_default_model():
    try:
        model = load_model("dl_churn_model.h5")
        return model
    except Exception as e:
        st.error("❌ Failed to load default DL model from repo.")
        st.stop()

loaded_model = load_default_model()

# If user uploads an alternate DL model, override the default
if uploaded_model is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            tmp.write(uploaded_model.read())
            tmp_path = tmp.name
        loaded_model = load_model(tmp_path)
        st.sidebar.success("✅ Using uploaded DL model.")
    except Exception as e:
        st.sidebar.error("❌ Failed to load uploaded model. Using default instead.")

# -----------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    ["🔮 Single Prediction", "📄 Batch Prediction (CSV)", "ℹ About"]
)

# -----------------------------------------------------------
# UNIVERSAL PREDICTION FUNCTION (DL ONLY)
# -----------------------------------------------------------

def make_prediction(df):
    """
    df must already be:
    - cleaned
    - encoded
    - reindexed to dl_features
    """
    preds = loaded_model.predict(df)
    prob = float(preds[0][0])
    pred_class = 1 if prob >= 0.5 else 0
    return pred_class, prob

# -----------------------------------------------------------
# 🔮 SINGLE PREDICTION
# -----------------------------------------------------------

if page == "🔮 Single Prediction":

    st.header("🔮 Predict Customer Churn (Deep Learning Model)")

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
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
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

        # Clean numeric-type issues
        df = clean_dataframe(df)

        # Apply encoders to categorical columns
        for col, encoder in encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col].astype(str))

        # Ensure all features exist and are in correct order
        for col in dl_features:
            if col not in df.columns:
                df[col] = 0  # safe default for missing columns

        df = df[dl_features]

        # Make prediction
        pred, prob = make_prediction(df)

        st.success(f"Prediction: {'Churn' if pred == 1 else 'No Churn'}")
        st.info(f"Churn Probability: {prob:.4f}")

# -----------------------------------------------------------
# 📄 BATCH PREDICTION
# -----------------------------------------------------------

elif page == "📄 Batch Prediction (CSV)":

    st.header("📄 Batch Churn Prediction (CSV) - Deep Learning Model")

    uploaded_csv = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
        st.write("Preview of uploaded data:", df.head())

        # Drop customerID if present, to be safe
        if "customerID" in df.columns:
            df = df.drop("customerID", axis=1)

        if st.button("Run Batch Prediction"):

            df_clean = clean_dataframe(df.copy())

            # Apply encoders where applicable
            for col, encoder in encoders.items():
                if col in df_clean.columns:
                    df_clean[col] = encoder.transform(df_clean[col].astype(str))

            # Ensure all DL features exist (add any missing as 0)
            for col in dl_features:
                if col not in df_clean.columns:
                    df_clean[col] = 0

            # Reorder columns
            df_clean = df_clean[dl_features]

            # Predict in batch
            preds_raw = loaded_model.predict(df_clean)
            probs = preds_raw.flatten()
            preds = (probs >= 0.5).astype(int)

            df["Churn_Pred"] = preds
            df["Churn_Prob"] = probs

            st.success("✅ Batch prediction completed!")
            st.dataframe(df)

            st.download_button(
                "Download Results as CSV",
                df.to_csv(index=False).encode("utf-8"),
                "churn_predictions_dl.csv",
                "text/csv"
            )

# -----------------------------------------------------------
# ℹ ABOUT
# -----------------------------------------------------------

else:
    st.header("ℹ About This App")
    st.write("""
    This app predicts telecom customer churn using a **Deep Learning model** trained on the Telco Customer Churn dataset.

    **Key details:**
    - `customerID` is removed from training and prediction
    - Categorical features are LabelEncoded using the same encoders as training
    - SMOTE is used to handle class imbalance
    - Default DL model is loaded from the GitHub repo
    - You can optionally upload a different `.h5` DL model (same feature schema)

    Pages:
    - 🔮 Single Prediction: Manually enter customer details
    - 📄 Batch Prediction: Upload a CSV for multiple customers
    """)
