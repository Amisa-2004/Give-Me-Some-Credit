import streamlit as st
import numpy as np
import joblib
import time
import requests
from streamlit_lottie import st_lottie

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="centered"
)

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("credit_pipeline.pkl")

model = load_model()

# ---------------------------------------------------
# Load Lottie Animation
# ---------------------------------------------------
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottie(
    "https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json"
)

# ---------------------------------------------------
# Custom Styling
# ---------------------------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Header
# ---------------------------------------------------
st.title("💳 Credit Default Risk Prediction")
st.caption("AI Model (Random Forest | ROC-AUC ≈ 0.83)")
st.markdown("---")

# Animation
if lottie_ai:
    st_lottie(lottie_ai, height=180)

# ---------------------------------------------------
# Input Section
# ---------------------------------------------------
st.subheader("📋 Enter Borrower Details")

col1, col2 = st.columns(2)

with col1:
    revolving_util = st.slider(
        "Revolving Utilization",
        0.0, 1.5, 0.3,
        help="Credit card usage ratio"
    )
    age = st.slider("Age", 18, 100, 35)
    debt_ratio = st.slider("Debt Ratio", 0.0, 2.0, 0.5)
    monthly_income = st.number_input(
        "Monthly Income ($)",
        0.0, 50000.0, 4000.0, step=500.0
    )
    dependents = st.slider("Dependents", 0, 10, 1)

with col2:
    past_due_30_59 = st.slider("30–59 Days Past Due", 0, 10, 0)
    past_due_60_89 = st.slider("60–89 Days Past Due", 0, 10, 0)
    past_due_90 = st.slider("90 Days Late", 0, 10, 0)
    open_credit = st.slider("Open Credit Lines", 0, 20, 5)
    real_estate_loans = st.slider("Real Estate Loans", 0, 10, 1)

st.markdown("---")

# ---------------------------------------------------
# Prediction Section
# ---------------------------------------------------
if st.button("🔍 Predict Risk"):

    with st.spinner("Analyzing borrower profile... 🤖"):
        input_data = np.array([[revolving_util, age, past_due_30_59,
                                debt_ratio, monthly_income, open_credit,
                                past_due_90, real_estate_loans,
                                past_due_60_89, dependents]])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        time.sleep(1)  # small delay for animation effect

    st.subheader("📊 Risk Assessment")

    # Smooth animated progress bar
    progress_bar = st.progress(0)
    for percent in np.linspace(0, probability, 50):
        progress_bar.progress(float(percent))
        time.sleep(0.01)

    st.write(f"### Default Probability: **{probability:.2%}**")

    # Risk Category Display
    if probability < 0.3:
        st.balloons()
        st.success("🟢 LOW RISK — Safe Borrower")

    elif probability < 0.6:
        st.warning("🟡 MEDIUM RISK — Monitor Closely")

    else:
        st.snow()
        st.error("🔴 HIGH RISK — Likely to Default")

    st.info("""
    🔎 Interpretation Guide:
    - Below 30% → Low Risk
    - 30%–60% → Medium Risk
    - Above 60% → High Risk
    """)

st.markdown("---")
