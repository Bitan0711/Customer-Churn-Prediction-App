import streamlit as st
import pandas as pd
import numpy as np
import pickle
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
# LOAD MODELS / ENCODERS (ML OR DL)
# -----------------------------------------------------------

st.sidebar.title("⚙️ Settings")

uploaded_model = st.sidebar.file_uploader("Upload Model (.pk1/.pkl/.h5)", type=["pk1", "pkl", "h5"])
uploaded_encoders = st.sidebar.file_uploader("Upload Encoders (.pk1/.pkl)", type=["pk1", "pkl"])

def load_any_model(file):
    filename = file.name.lower()
    if filename.endswith(".h5"):
        return load_model(file), "DL"
    else:
        return pickle.load(file), "ML"

loaded_model = None
model_type = None
encoders = None
feature_names = None

# Try uploaded models first
if uploaded_model and uploaded_encoders:
    loaded_model, model_type = load_any_model(uploaded_model)
    encoders = pickle.load(uploaded_encoders)

else:
    try:
        # Default ML model
        with open("customer_churn_model.pk1", "rb") as f:
            model_data = pickle.load(f)
        loaded_model = model_data["model"]
        feature_names = model_data["features_name"]
        model_type = "ML"

        with open("encoders.pk1", "rb") as f:
            encoders = pickle.load(f)

    except:
        st.warning("⚠ No default model/encoders found. Upload in sidebar.")

if uploaded_model and model_type == "ML":
    feature_names = loaded_model["features_name"] if isinstance(loaded_model, dict) else None

# -----------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    ["🔮 Single Prediction", "📄 Batch Prediction (CSV)", "ℹ About"]
)

# -----------------------------------------------------------
# UNIVERSAL PREDICTION HANDLER (ML + DL)
# -----------------------------------------------------------

def make_prediction(df):
    """
    Works with both:
    - ML models (RandomForest, XGBoost)
    - DL Keras models (.h5)
    """

    if model_type == "DL":
        prob = float(loaded_model.predict(df)[0][0])
        pred = 1 if prob >= 0.5 else 0
        return pred, prob

    else:  # ML model
        pred = loaded_model.predict(df)[0]
        prob = loaded_model.predict_proba(df)[0][1]
        return pred, prob

# -----------------------------------------------------------
# 🔮 SINGLE PREDICTION
# -----------------------------------------------------------

if page == "🔮 Single Prediction":

    st.header("🔮 Predict Customer Churn")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure", 0, 72)

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
        if not loaded_model or not encoders:
            st.error("Model/Encoders missing.")
        else:
            row = {
                'gender': gender, 'SeniorCitizen': SeniorCitizen,
                'Partner': Partner, 'Dependents': Dependents,
                'tenure': tenure, 'PhoneService': PhoneService,
                'MultipleLines': MultipleLines, 'InternetService': InternetService,
                'OnlineSecurity': OnlineSecurity, 'OnlineBackup': OnlineBackup,
                'DeviceProtection': DeviceProtection, 'TechSupport': TechSupport,
                'StreamingTV': StreamingTV, 'StreamingMovies': StreamingMovies,
                'Contract': Contract, 'PaperlessBilling': PaperlessBilling,
                'PaymentMethod': PaymentMethod, 'MonthlyCharges': MonthlyCharges,
                'TotalCharges': TotalCharges
            }

            df = pd.DataFrame([row])
            df = clean_dataframe(df)

            for col, encoder in encoders.items():
                df[col] = encoder.transform(df[col].astype(str))

            if model_type == "ML":
                df = df[feature_names]

            pred, prob = make_prediction(df)

            st.success(f"Prediction: {'Churn' if pred == 1 else 'No Churn'}")
            st.info(f"Probability: {prob:.4f}")

# -----------------------------------------------------------
# 📄 CSV BATCH PREDICTION
# -----------------------------------------------------------

elif page == "📄 Batch Prediction (CSV)":

    st.header("📄 Upload CSV for Batch Prediction")

    uploaded_csv = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.write("Preview:", df.head())

        if st.button("Run Batch Prediction"):

            df_clean = clean_dataframe(df.copy())

            for col, encoder in encoders.items():
                if col in df_clean.columns:
                    df_clean[col] = encoder.transform(df_clean[col].astype(str))

            if model_type == "ML":
                df_clean = df_clean[feature_names]

            preds = []
            probs = []

            for i in range(len(df_clean)):
                pred, prob = make_prediction(df_clean.iloc[i:i+1])
                preds.append(pred)
                probs.append(prob)

            df["Churn_Pred"] = preds
            df["Churn_Prob"] = probs

            st.success("Batch prediction completed!")
            st.dataframe(df)

            st.download_button(
                "Download Results",
                df.to_csv(index=False).encode("utf-8"),
                "predictions.csv",
                "text/csv"
            )

# -----------------------------------------------------------
# ℹ ABOUT
# -----------------------------------------------------------

else:
    st.header("ℹ About This App")
    st.write("""
    This app predicts telecom customer churn using either:

    **Machine Learning models**
    - Random Forest  
    - XGBoost  
    - Logistic Regression  

    **Deep Learning models**
    - TensorFlow/Keras (.h5)

    Features:
    ✔ Single Prediction  
    ✔ Batch CSV Prediction  
    ✔ Upload custom ML/DL model  
    ✔ Automatic Encoding  
    ✔ Dark UI Theme  
    """)
