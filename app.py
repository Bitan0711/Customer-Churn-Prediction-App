import streamlit as st
import pandas as pd
import numpy as np
import pickle

def clean_dataframe(df):
    # Fix TotalCharges missing values
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].replace(" ", "0.0")
        df["TotalCharges"] = df["TotalCharges"].fillna("0.0")
        df["TotalCharges"] = df["TotalCharges"].astype(float)

    # Fix MonthlyCharges if needed
    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)

    # Fix tenure if needed
    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)

    return df
#------ cleaning csv ------




st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# -----------------------------------------------------------
# THEME + CSS SYSTEM
# -----------------------------------------------------------

def set_theme(dark=False):
    if dark:
        css = """
        <style>
        /* MAIN APP BACKGROUND */
        .stApp {
            background-color: #0f0f0f !important;
            color: #f2f2f2 !important;
        }

        /* MAIN PAGE WRAPPER */
        .stAppViewContainer, .stMain {
            background-color: #0f0f0f !important;
        }

        /* SIDEBAR */
        section[data-testid="stSidebar"]{
            background-color: #1a1a1a !important;
        }

        /* HEADERS */
        h1, h2, h3, h4, h5, h6, label, p, span, div {
            color: #e6e6e6 !important;
        }

        /* BUTTONS */
        .stButton>button {
            background-color: #444 !important;
            color: white !important;
            border-radius: 8px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #666 !important;
        }

        /* INPUT FIELDS */
        input, select, textarea {
            background-color: #333 !important;
            color: #fff !important;
            border-radius: 6px !important;
        }

        /* REMOVE THE WHITE/BLACK BAR UNDER HEADER */
        .block-container {
            padding-top: 0 !important;
            background-color: inherit !important;
        }
        </style>
        """
    else:
        css = """
        <style>
        .stApp {
            background-color: #f5f7fa !important;
            color: #333 !important;
        }

        .stAppViewContainer, .stMain {
            background-color: #f5f7fa !important;
        }

        section[data-testid="stSidebar"]{
            background-color: #ffffff !important;
        }

        h1, h2, h3, h4, h5, h6, label {
            color: #222 !important;
        }

        .stButton>button {
            background-color: #4CAF50 !important;
            color: white !important;
            border-radius: 8px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049 !important;
        }

        input, select, textarea {
            background-color: #ffffff !important;
            color: #000000 !important;
            border-radius: 6px !important;
        }

        .block-container {
            padding-top: 0 !important;
            background-color: inherit !important;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)
with st.sidebar:
    st.title("⚙️ Settings")
    theme_choice = st.radio("Theme", ["Light", "Dark"])

set_theme(dark=(theme_choice == "Dark"))


# -----------------------------------------------------------
# PROFESSIONAL DASHBOARD HEADER
# -----------------------------------------------------------

st.markdown("""
<div style='padding: 0 0 15px 0;'>
    <h1 style='font-weight:700; margin-bottom: 0;'>📊 Customer Churn Prediction Dashboard</h1>
    <p style='font-size:17px; margin-top:5px;'>Predict customer churn using ML.</p>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# LOAD MODEL / ENCODERS (local or uploaded)
# -----------------------------------------------------------

st.sidebar.subheader("📁 Upload Model Files (Optional)")

uploaded_model = st.sidebar.file_uploader("Upload Model File (.pk1)", type=["pk1", "pkl"])
uploaded_encoders = st.sidebar.file_uploader("Upload Encoders File (.pk1)", type=["pk1", "pkl"])


def load_pickle(file):
    return pickle.load(file)


model_data = None
encoders = None

# FIRST PRIORITY → Uploaded files
if uploaded_model and uploaded_encoders:
    model_data = load_pickle(uploaded_model)
    encoders = load_pickle(uploaded_encoders)

# SECOND PRIORITY → Load from repo
else:
    try:
        with open("customer_churn_model.pk1", "rb") as f:
            model_data = pickle.load(f)
        with open("encoders.pk1", "rb") as f:
            encoders = pickle.load(f)
    except:
        st.warning("⚠ Model and Encoders not found. Please upload the pickle files.")

if model_data:
    loaded_model = model_data["model"]
    feature_names = model_data["features_name"]


# -----------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------

page = st.sidebar.radio(
    "Navigate",
    ["🔮 Single Prediction", "📄 Batch Prediction (CSV)", "ℹ About"]
)

# -----------------------------------------------------------
# 🔮 SINGLE PREDICTION PAGE
# -----------------------------------------------------------

if page == "🔮 Single Prediction":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔮 Predict Churn for a Single Customer")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (Months)", 0, 72, 12)

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
        PaymentMethod = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )

    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, MonthlyCharges * tenure)

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔍 Predict"):
        if not model_data or not encoders:
            st.error("❌ Model/Encoders missing.")
        else:
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


            # Apply encoders
            for col, enc in encoders.items():
                df[col] = enc.transform(df[col].astype(str))

            df = df[feature_names]

            pred = loaded_model.predict(df)[0]
            prob = loaded_model.predict_proba(df)[0][1]

            st.success(f"Prediction: **{'Churn' if pred==1 else 'No Churn'}**")
            st.info(f"Churn Probability: **{prob:.4f}**")


# -----------------------------------------------------------
# 📄 BATCH PREDICTION PAGE
# -----------------------------------------------------------

elif page == "📄 Batch Prediction (CSV)":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📄 Upload CSV for Batch Predictions")
    st.write("CSV should contain customer records without the **Churn** column.")

    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.write("Preview:", df.head())

        if st.button("🚀 Run Batch Prediction"):
            if not encoders or not model_data:
                st.error("❌ Missing model/encoders.")
            else:
                df_encoded = df.copy()
                df_encoded = clean_dataframe(df_encoded)


                for col, encoder in encoders.items():
                    if col in df_encoded.columns:
                        df_encoded[col] = encoder.transform(df_encoded[col].astype(str))

                df_encoded = df_encoded[feature_names]

                preds = loaded_model.predict(df_encoded)
                probs = loaded_model.predict_proba(df_encoded)[:, 1]

                df["Churn_Pred"] = preds
                df["Churn_Prob"] = probs

                st.success("Batch prediction complete!")
                st.dataframe(df)

                st.download_button(
                    "Download Predictions CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    "predictions.csv",
                    "text/csv"
                )

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------
# ABOUT PAGE
# -----------------------------------------------------------

else:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("ℹ About This App")
    st.write("""
    This Machine Learning dashboard predicts customer churn using a trained 
    Random Forest Classifier.

    ### Features:
    - Single customer prediction  
    - Bulk CSV prediction  
    - Upload custom model files  
    - Automatic encoding  
    - Modern UI + Theme switching  

    Created by **Bitan**.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
