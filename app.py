"""
╔══════════════════════════════════════════════════╗
║   Give Me Some Credit — Streamlit UI             ║
║   Run: streamlit run credit_risk_app.py          ║
╚══════════════════════════════════════════════════╝
Requirements:
    pip install streamlit scikit-learn pandas numpy plotly
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditLens — Risk Assessment",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────
# Custom CSS — Dark Finance Aesthetic
# ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #080c14;
    color: #e8eaf0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1525 0%, #0a1020 100%);
    border-right: 1px solid #1e2d47;
}
[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

/* ── Main area ── */
.main .block-container {
    padding: 2rem 3rem;
    max-width: 1200px;
}

/* ── Cards ── */
.metric-card {
    background: linear-gradient(135deg, #0f1e35 0%, #0d1828 100%);
    border: 1px solid #1e3050;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00d4ff, #7b61ff, #ff4d6d);
}

/* ── Hero Title ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4ff 0%, #7b61ff 50%, #ff4d6d 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    line-height: 1.1;
    margin: 0;
}
.hero-sub {
    color: #6e82a0;
    font-size: 1rem;
    font-weight: 400;
    letter-spacing: 0.05em;
    margin-top: 0.5rem;
}

/* ── Risk Badge ── */
.risk-badge-low {
    background: linear-gradient(135deg, #003d2b, #005c3f);
    border: 1px solid #00c27a;
    color: #00ff9d;
    border-radius: 50px;
    padding: 0.4rem 1.2rem;
    font-weight: 600;
    font-size: 0.9rem;
    display: inline-block;
}
.risk-badge-medium {
    background: linear-gradient(135deg, #3d2e00, #5c4600);
    border: 1px solid #c2900a;
    color: #ffc107;
    border-radius: 50px;
    padding: 0.4rem 1.2rem;
    font-weight: 600;
    font-size: 0.9rem;
    display: inline-block;
}
.risk-badge-high {
    background: linear-gradient(135deg, #3d0010, #5c0018);
    border: 1px solid #c2002a;
    color: #ff4d6d;
    border-radius: 50px;
    padding: 0.4rem 1.2rem;
    font-weight: 600;
    font-size: 0.9rem;
    display: inline-block;
}

/* ── Section header ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4a9eff;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e3050;
}

/* ── Sliders & inputs ── */
.stSlider > div > div > div > div { background: #00d4ff !important; }
.stSlider [data-baseweb="slider"] { padding: 0.5rem 0; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0066cc, #7b61ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.05em !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0, 102, 204, 0.4) !important;
}

/* ── Dividers ── */
hr { border-color: #1e3050; }

/* ── Info boxes ── */
.stInfo { background: #0d1e35; border-left-color: #00d4ff; }

/* ── Number inputs ── */
.stNumberInput input { background: #0f1e35; border-color: #1e3050; color: #e8eaf0; }

/* ── Select boxes ── */
.stSelectbox select { background: #0f1e35; border-color: #1e3050; color: #e8eaf0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("credit_pipeline.pkl")

model = load_model()

FEATURE_NAMES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfDependents",
    "NumberOfTime60-89DaysPastDueNotWorse",
]


# ─────────────────────────────────────────────────
# Gauge Chart
# ─────────────────────────────────────────────────
def make_gauge(probability):
    color = "#00ff9d" if probability < 0.35 else ("#ffc107" if probability < 0.50 else "#ff4d6d")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability * 100, 1),
        number={"suffix": "%", "font": {"size": 48, "color": color, "family": "Syne"}},
        title={"text": "Default Probability", "font": {"size": 14, "color": "#6e82a0", "family": "Space Grotesk"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#2a3f5f", "tickfont": {"color": "#4a6080"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#0a1020",
            "bordercolor": "#1e3050",
            "steps": [
                {"range": [0, 30],  "color": "#003d2b"},
                {"range": [30, 50], "color": "#2d2000"},
                {"range": [50, 100],"color": "#3d0010"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": probability * 100,
            },
        },
    ))
    fig.update_layout(
        height=280,
        margin=dict(t=30, b=10, l=30, r=30),
        paper_bgcolor="#0d1525",
        font_color="#e8eaf0",
    )
    return fig


# ─────────────────────────────────────────────────
# Feature Importance Chart
# ─────────────────────────────────────────────────
def make_feature_chart(values, feature_labels):
    colors = ["#00d4ff", "#7b61ff", "#ff4d6d", "#00ff9d", "#ffc107",
              "#ff6b35", "#4ecdc4", "#45b7d1", "#96ceb4", "#ff9999"]
    fig = go.Figure(go.Bar(
        x=list(values),
        y=feature_labels,
        orientation='h',
        marker=dict(color=colors[:len(values)], line=dict(color="#1e3050", width=1)),
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(color="#6e82a0", size=11),
    ))
    fig.update_layout(
        height=320,
        margin=dict(t=10, b=10, l=10, r=60),
        paper_bgcolor="#0d1525",
        plot_bgcolor="#0d1525",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, color="#2a3f5f"),
        yaxis=dict(showgrid=False, color="#8a9ab0", tickfont=dict(size=11)),
        font=dict(family="Space Grotesk"),
    )
    return fig


# ─────────────────────────────────────────────────
# Radar Chart
# ─────────────────────────────────────────────────
def make_radar(normalized_vals, labels):
    fig = go.Figure(go.Scatterpolar(
        r=normalized_vals + [normalized_vals[0]],
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor='rgba(123,97,255,0.15)',
        line=dict(color='#7b61ff', width=2),
        marker=dict(color='#00d4ff', size=6),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0a1020",
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#2a3f5f"), gridcolor="#1e3050"),
            angularaxis=dict(tickfont=dict(size=10, color="#6e82a0"), gridcolor="#1e3050"),
        ),
        height=300,
        margin=dict(t=30, b=30, l=50, r=50),
        paper_bgcolor="#0d1525",
        font=dict(family="Space Grotesk"),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────
# SIDEBAR — Input Panel
# ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="hero-title" style="font-size:1.8rem;">CreditLens</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">AI-Powered Credit Risk Assessment</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p class="section-header">👤 Personal Profile</p>', unsafe_allow_html=True)
    age = st.slider("Age", min_value=18, max_value=100, value=42, step=1,
                    help="Applicant's age in years")
    num_dependents = st.slider("Number of Dependents", min_value=0, max_value=20, value=0,
                               help="Number of dependents (family members)")

    st.markdown("---")
    st.markdown('<p class="section-header">💰 Financial Health</p>', unsafe_allow_html=True)
    monthly_income = st.number_input("Monthly Income (USD)", min_value=0, max_value=500000,
                                     value=5400, step=100,
                                     help="Gross monthly income in USD")
    revolving_util = st.slider("Revolving Credit Utilization", min_value=0.0, max_value=1.0,
                                value=0.15, step=0.01, format="%.2f",
                                help="Ratio of outstanding balance to credit limit (0 = none, 1 = maxed out)")
    debt_ratio = st.slider("Debt Ratio", min_value=0.0, max_value=1.0,
                           value=0.35, step=0.01, format="%.2f",
                           help="Monthly debt payments / monthly gross income")

    st.markdown("---")
    st.markdown('<p class="section-header">🏦 Credit Lines</p>', unsafe_allow_html=True)
    num_open_credit = st.slider("Open Credit Lines & Loans", min_value=0, max_value=60, value=8,
                                help="Number of open credit lines and loans")
    num_real_estate = st.slider("Real Estate Loans or Lines", min_value=0, max_value=20, value=1,
                                help="Number of mortgage and real estate loans")

    st.markdown("---")
    st.markdown('<p class="section-header">⚠️ Delinquency History</p>', unsafe_allow_html=True)
    late_30_59 = st.slider("30–59 Days Late (past 2 years)", min_value=0, max_value=20, value=0,
                           help="Times 30–59 days past due (not worse)")
    late_60_89 = st.slider("60–89 Days Late (past 2 years)", min_value=0, max_value=20, value=0,
                           help="Times 60–89 days past due (not worse)")
    late_90 = st.slider("90+ Days Late (past 2 years)", min_value=0, max_value=20, value=0,
                        help="Times 90+ days past due")

    st.markdown("---")
    predict_btn = st.button("🔍 Analyze Credit Risk", use_container_width=True)


# ─────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown('<h1 class="hero-title">Credit Risk Intelligence</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">POWERED BY GRADIENT BOOSTING · GIVE ME SOME CREDIT DATASET</p>', unsafe_allow_html=True)

st.markdown("---")

# ─── Default state (pre-prediction) ───
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ─── Run Prediction ───
features = [revolving_util, age, late_30_59, debt_ratio,
            monthly_income, num_open_credit, late_60_89,
            num_real_estate, num_dependents, late_90]

if predict_btn:
    X_input = np.array(features).reshape(1, -1)
    prob = model.predict_proba(X_input)[0][1]
    pred = model.predict(X_input)[0]
    st.session_state.last_prediction = {"prob": prob, "pred": pred, "features": features}

# ─── Display Results ───
if st.session_state.last_prediction:
    result = st.session_state.last_prediction
    prob = result["prob"]
    pred = result["pred"]

    # Risk tier
    if prob < 0.35:
        risk_tier = "LOW RISK"
        badge_class = "risk-badge-low"
        risk_color = "#00ff9d"
        risk_emoji = "✅"
        recommendation = "Applicant demonstrates strong financial discipline. Likely to honor commitments."
    elif prob < 0.50:
        risk_tier = "MODERATE RISK"
        badge_class = "risk-badge-medium"
        risk_color = "#ffc107"
        risk_emoji = "⚠️"
        recommendation = "Some risk factors present. Consider additional verification or adjusted terms."
    else:
        risk_tier = "HIGH RISK"
        badge_class = "risk-badge-high"
        risk_color = "#ff4d6d"
        risk_emoji = "🚨"
        recommendation = "Significant delinquency or financial stress indicators detected. Exercise caution."

    # ── Row 1: Gauge + Key Metrics ──
    col_gauge, col_metrics = st.columns([1, 2])

    with col_gauge:
        st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
        st.plotly_chart(make_gauge(prob), use_container_width=True, config={"displayModeBar": False})
        st.markdown(f'<div style="text-align:center; margin-top:-1rem;">'
                    f'<span class="{badge_class}">{risk_emoji} {risk_tier}</span>'
                    f'</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_metrics:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"#### {risk_emoji} Assessment Summary")
        st.markdown(f"> *{recommendation}*")
        st.markdown("---")

        m1, m2, m3 = st.columns(3)
        m1.metric("Default Probability", f"{prob*100:.1f}%",
                  delta=f"{'▲ Above' if prob > 0.5 else '▼ Below'} threshold")
        m2.metric("Credit Utilization", f"{revolving_util*100:.0f}%",
                  delta="High" if revolving_util > 0.5 else "Healthy")
        m3.metric("Debt Ratio", f"{debt_ratio*100:.0f}%",
                  delta="High" if debt_ratio > 0.43 else "Acceptable")

        m4, m5, m6 = st.columns(3)
        m4.metric("Age", str(age))
        m5.metric("Monthly Income", f"${monthly_income:,.0f}")
        total_late = late_30_59 + late_60_89 + late_90
        m6.metric("Total Late Payments", str(total_late),
                  delta="⚠️ Concerning" if total_late > 2 else "Clean history")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 2: Feature Importance + Radar ──
    col_imp, col_radar = st.columns(2)

    with col_imp:
        st.markdown('<p class="section-header">📊 Feature Contributions</p>', unsafe_allow_html=True)
        importances = model.steps[-1][1].feature_importances_
        feat_labels = [
            "Credit Utilization", "Age", "30-59d Late", "Debt Ratio",
            "Monthly Income", "Open Credit Lines", "90d+ Late",
            "Real Estate Loans", "Dependents", "60-89d Late"
        ]
        fig_imp = make_feature_chart(importances, feat_labels)
        st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})

    with col_radar:
        st.markdown('<p class="section-header">🕸️ Risk Profile Radar</p>', unsafe_allow_html=True)
        # Normalize each feature for radar display
        radar_vals = [
            min(revolving_util, 1.0),
            min(late_30_59 / 5, 1.0),
            min(late_60_89 / 5, 1.0),
            min(late_90 / 5, 1.0),
            min(debt_ratio, 1.0),
            min(num_dependents / 10, 1.0),
        ]
        radar_labels = ["Credit Util.", "30-59d Late", "60-89d Late", "90d+ Late", "Debt Ratio", "Dependents"]
        fig_radar = make_radar(radar_vals, radar_labels)
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")

    # ── Row 3: Detailed Breakdown Table ──
    st.markdown('<p class="section-header">📋 Input Summary & Risk Flags</p>', unsafe_allow_html=True)

    def flag(val, threshold, high="🔴 High", ok="🟢 OK"):
        return high if val > threshold else ok

    summary_data = {
        "Feature": feat_labels,
        "Your Value": [
            f"{revolving_util:.2f}", str(age), str(late_30_59), f"{debt_ratio:.2f}",
            f"${monthly_income:,.0f}", str(num_open_credit), str(late_90),
            str(num_real_estate), str(num_dependents), str(late_60_89)
        ],
        "Status": [
            flag(revolving_util, 0.5),
            "🟢 OK" if 25 <= age <= 70 else "🟡 Note",
            flag(late_30_59, 1),
            flag(debt_ratio, 0.43),
            "🔴 Low" if monthly_income < 2000 else "🟢 OK",
            "🟡 High" if num_open_credit > 15 else "🟢 OK",
            flag(late_90, 0),
            "🟢 OK",
            "🟡 Many" if num_dependents > 5 else "🟢 OK",
            flag(late_60_89, 0),
        ],
        "Model Weight": [f"{v:.3f}" for v in importances],
    }

    df = pd.DataFrame(summary_data)
    st.dataframe(df, use_container_width=True, hide_index=True,
                 column_config={
                     "Feature": st.column_config.TextColumn("Feature", width="medium"),
                     "Your Value": st.column_config.TextColumn("Your Value", width="small"),
                     "Status": st.column_config.TextColumn("Status", width="small"),
                     "Model Weight": st.column_config.ProgressColumn("Model Importance", min_value=0, max_value=0.4),
                 })

else:
    # ── Welcome State ──
    st.markdown("""
    <div style="text-align:center; padding: 5rem 2rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">💳</div>
        <h2 style="font-family:'Syne',sans-serif; font-size:2rem; color:#4a9eff;">
            Ready to Assess Credit Risk
        </h2>
        <p style="color:#6e82a0; max-width:500px; margin: 1rem auto; line-height:1.8;">
            Fill in the applicant's financial profile in the sidebar on the left,
            then click <strong style="color:#7b61ff;">Analyze Credit Risk</strong>
            to get an instant AI-powered risk assessment with detailed breakdown.
        </p>
        <div style="display:flex; gap:2rem; justify-content:center; margin-top:3rem; flex-wrap:wrap;">
    """, unsafe_allow_html=True)

    for icon, label, desc in [
        ("📊", "Feature Analysis", "10 financial indicators evaluated"),
        ("🎯", "Risk Score", "Probability of default (0–100%)"),
        ("🕸️", "Risk Radar", "Visual profile of risk dimensions"),
        ("🏷️", "Risk Tier", "Low / Moderate / High classification"),
    ]:
        st.markdown(f"""
        <div class="metric-card" style="display:inline-block; width:200px; margin:0.5rem;">
            <div style="font-size:2rem;">{icon}</div>
            <div style="font-weight:600; color:#e8eaf0; margin-top:0.5rem;">{label}</div>
            <div style="color:#6e82a0; font-size:0.85rem; margin-top:0.3rem;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#2a3f5f; font-size:0.8rem; padding: 1rem 0;">
    CreditLens · Built on the <em>Give Me Some Credit</em> dataset ·
    Model: Gradient Boosting Classifier · For demonstration purposes only
</div>
""", unsafe_allow_html=True)