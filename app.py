import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# -----------------------------------------------------------
# BLACK THEME — CUSTOM CSS (YOU CAN EDIT COLORS HERE)
# -----------------------------------------------------------

def load_css():
    custom_css = """    
    /* ===== DARK THEME ===== */

    .stApp {
        background-color: #0e0e0e !important;
        color: #f1f1f1 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6, label {
        color: #ffffff !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #444444 !important;
        color: white !important;
        padding: 8px 18px !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
    }

    .stButton > button:hover {
        background-color: #666666 !important;
    }

    /* Input Fields */
    input, select, textarea {
        background-color: #222 !important;
        color: white !important;
        border-radius: 6px !important;
        border: 1px solid #444 !important;
    }

    /* Dataframe background */
    .dataframe {
        color: white !important;
    }
    """
    st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

load_css()

# -----------------------------------------------------------
# CLEANING FUNCTION (Fix CSV TotalCharges issues)
# -----------------------------------------------------------

def clean_dataframe(df):
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].replace(" ", "0.0")
        df["TotalCharges"] = df["TotalCharges"].fillna("0.0")
        df["TotalCharges"] = df["TotalCharges"].astype(float)

    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)

    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)

    return df

# -----------------------------------------------------------
# LOAD MODEL / ENCODERS
# -----------------------------------------------------------

st.sidebar.title("⚙️ Settings")

uploaded_model = st.sidebar.file_uploader("Upload Model (.pk1)", type=["pk1", "pkl"])
uploaded_encoders = st.sidebar.file_uploader("Upload Encoders (.pk1)", type=["pk1", "pkl"])

def load_pickle(file):
    return pickle.load(file)

model_data = None
encoders = None

if uploaded_model and uploaded_encoders:
    model_data = load_pickle(uploaded_model)
    encoders = load_pickle(uploaded_encoders)
else:
    try:
        with open("customer_churn_model.pk1", "rb") as f:
            model_data = pickle.load(f)
        with open("encoders.pk1", "rb") as f:
            encoders = pickle.load(f)
    except:
        st.warning("⚠ Model/Encoders not found. Upload in sidebar.")

if model_data:
    loaded_model = model_data["model"]
    feature_names = model_data["features_name"]

# -----------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    ["🔮 Single Prediction", "📄 Batch Prediction (CSV)", "ℹ About"]
)

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
        if not model_data or not encoders:
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

            df = df[feature_names]

            pred = loaded_model.predict(df)[0]
            prob = loaded_model.predict_proba(df)[0][1]

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

            df_clean = df_clean[feature_names]

            preds = loaded_model.predict(df_clean)
            probs = loaded_model.predict_proba(df_clean)[:, 1]

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
    This ML app predicts customer churn using a trained Random Forest model.

    Features:
    - Single Prediction  
    - Batch CSV Prediction  
    - Upload your own model  
    - Customizable CSS  
    - Dark Mode UI  
    """)

