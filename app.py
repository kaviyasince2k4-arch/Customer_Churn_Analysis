import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load saved model
# -----------------------------
model = joblib.load("models/logistic_regression_churn.pkl")
scaler = joblib.load("models/churn_scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# -----------------------------
# Header
# -----------------------------
st.title("📊 Customer Churn Prediction")
st.markdown(
    "### Predict whether a customer is likely to churn "
    "using Machine Learning."
)

st.divider()


# -----------------------------
# Customer Information
# -----------------------------
st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

with col2:
    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )
    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col3:
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )


st.subheader("💳 Billing Information")

col4, col5, col6 = st.columns(3)

with col4:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col5:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

with col6:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=500.0
)


st.divider()


# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Customer Churn", use_container_width=True):

    customer = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    customer_df = pd.DataFrame([customer])

    # Encode categorical features
    customer_encoded = pd.get_dummies(
        customer_df,
        drop_first=True
    )

    # Match training features
    customer_encoded = customer_encoded.reindex(
        columns=feature_names,
        fill_value=0
    )

    # Scale features
    customer_scaled = scaler.transform(customer_encoded)

    # Prediction
    prediction = model.predict(customer_scaled)[0]
    probability = model.predict_proba(customer_scaled)[0][1]

    # -----------------------------
    # Display Result
    # -----------------------------
    st.divider()
    st.subheader("🎯 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if prediction == 1:
            st.error("⚠️ Customer is likely to churn")
        else:
            st.success("✅ Customer is likely to stay")

    with result_col2:

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

    # Business interpretation
    st.subheader("💡 Business Interpretation")

    if probability >= 0.5:
        st.warning(
            "This customer has a relatively high predicted churn "
            "probability. The company may consider customer-retention "
            "actions."
        )
    else:
        st.info(
            "This customer has a relatively low predicted churn "
            "probability based on the model."
        )