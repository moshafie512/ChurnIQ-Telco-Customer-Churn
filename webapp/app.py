from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ChurnIQ | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PATHS & MODEL LOADING
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "XGBoost Model"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "xgb_churn_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")

    return model, scaler, feature_columns


try:
    model, scaler, feature_columns = load_artifacts()
    MODEL_READY = True
except Exception as exc:
    MODEL_READY = False
    MODEL_ERROR = str(exc)



# =========================================================
# UI HELPERS (English-only, original dark theme)
# =========================================================

def tr(text):
    """Keep existing labels in English without language switching."""
    return text


def _save_churn_input(name):
    """Keep Churn Prediction inputs independently of page-widget lifecycle."""
    widget_key = f"cp_widget_{name}"
    if widget_key in st.session_state:
        st.session_state[f"cp_saved_{name}"] = st.session_state[widget_key]


def _churn_widget_state(name, default, allowed=None):
    """Restore a saved value whenever the Churn Prediction page is revisited."""
    saved = st.session_state.get(f"cp_saved_{name}", default)
    if allowed is not None and saved not in allowed:
        saved = default
        st.session_state[f"cp_saved_{name}"] = saved
    key = f"cp_widget_{name}"
    if key not in st.session_state:
        st.session_state[key] = saved
    elif allowed is not None and st.session_state[key] not in allowed:
        st.session_state[key] = default


def _save_retention_target_count():
    """Persist the campaign target count when its number input changes."""
    st.session_state.retention_target_count_saved = st.session_state.retention_target_count_widget


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    :root {
        --bg: #0B1120;
        --panel: #111B2E;
        --panel-light: #17243A;
        --border: #263750;
        --text: #F1F5F9;
        --muted: #94A3B8;
        --accent: #38BDF8;
        --green: #34D399;
        --red: #FB7185;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: #0E1728;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    .hero {
        padding: 30px 32px;
        border: 1px solid var(--border);
        border-radius: 22px;
        background:
            radial-gradient(
                circle at 90% 10%,
                rgba(56,189,248,0.15),
                transparent 35%
            ),
            linear-gradient(135deg, #142239, #10192B);
        margin-bottom: 25px;
    }

    .eyebrow {
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 11px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .hero h1 {
        color: var(--text);
        font-size: clamp(30px, 4vw, 43px);
        line-height: 1.15;
        margin: 0;
        font-weight: 800;
    }

    .hero p {
        color: var(--muted);
        font-size: 15px;
        line-height: 1.8;
        max-width: 760px;
        margin-top: 14px;
        margin-bottom: 0;
    }

    .section-title {
        font-size: 22px;
        font-weight: 750;
        color: var(--text);
        margin: 12px 0 5px 0;
    }

    .section-subtitle {
        color: var(--muted);
        font-size: 13px;
        margin-bottom: 20px;
    }

    .metric-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 20px;
        min-height: 125px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .metric-value {
        color: var(--text);
        font-size: 27px;
        font-weight: 800;
    }

    .metric-note {
        color: var(--muted);
        font-size: 11px;
        margin-top: 8px;
    }

    .result-card {
        padding: 25px;
        border-radius: 20px;
        background: var(--panel);
        border: 1px solid var(--border);
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .result-risk {
        color: var(--red);
        font-size: 28px;
        font-weight: 800;
    }

    .result-safe {
        color: var(--green);
        font-size: 28px;
        font-weight: 800;
    }

    .result-caption {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.8;
    }

    div[data-testid="stForm"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 22px;
    }

    .stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: var(--accent);
        color: #07111F;
        border: none;
        border-radius: 10px;
        font-weight: 800;
        min-height: 45px;
        transition: 0.2s ease;
    }

    .stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #7DD3FC;
        color: #07111F;
        border: none;
    }

    h1, h2, h3 {
        color: var(--text);
    }

    [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    hr {
        border-color: var(--border);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# HELPERS
# =========================================================

def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_heading(title, subtitle):
    st.markdown(
        f"""
        <div class="section-title">{title}</div>
        <div class="section-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def build_customer_features(data, feature_columns):
    """
    Reproduce the feature encoding used during model training.
    """

    row = {column: 0 for column in feature_columns}

    # Binary features
    binary_mapping = {
        "gender": 1 if data["gender"] == "Male" else 0,
        "SeniorCitizen": data["SeniorCitizen"],
        "Partner": 1 if data["Partner"] == "Yes" else 0,
        "Dependents": 1 if data["Dependents"] == "Yes" else 0,
        "PhoneService": 1 if data["PhoneService"] == "Yes" else 0,
        "PaperlessBilling": (
            1 if data["PaperlessBilling"] == "Yes" else 0
        ),
    }

    for column, value in binary_mapping.items():
        if column in row:
            row[column] = value

    # Tenure and monthly charges remain unscaled here.
    row["tenure"] = data["tenure"]
    row["MonthlyCharges"] = data["MonthlyCharges"]

    # During training, "No phone service" was replaced with "No".
    # Therefore, encode it as 0 rather than NaN.
    if data["PhoneService"] == "No":
        row["MultipleLines"] = 0
    else:
        row["MultipleLines"] = (
            1 if data["MultipleLines"] == "Yes" else 0
        )

    # Internet-related services
    internet_services = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    for column in internet_services:
        # During training, "No internet service" was replaced with "No".
        # Encode those cases as 0, matching the training preprocessing.
        if data["InternetService"] == "No":
            row[column] = 0
        else:
            row[column] = (
                1 if data[column] == "Yes" else 0
            )

    # One-hot encoding: InternetService
    if data["InternetService"] == "Fiber optic":
        row["InternetService_Fiber optic"] = 1
    elif data["InternetService"] == "No":
        row["InternetService_No"] = 1

    # One-hot encoding: Contract
    if data["Contract"] == "One year":
        row["Contract_One year"] = 1
    elif data["Contract"] == "Two year":
        row["Contract_Two year"] = 1

    # One-hot encoding: PaymentMethod
    payment_feature_map = {
        "Credit card (automatic)": (
            "PaymentMethod_Credit card (automatic)"
        ),
        "Electronic check": (
            "PaymentMethod_Electronic check"
        ),
        "Mailed check": (
            "PaymentMethod_Mailed check"
        ),
    }

    selected_payment_feature = payment_feature_map.get(
        data["PaymentMethod"]
    )

    if selected_payment_feature:
        row[selected_payment_feature] = 1

    # Ensure exact feature order.
    features = pd.DataFrame([row], columns=feature_columns)

    # Match the feature types expected by the model.
    return features.astype(float)


def scale_customer_features(features, scaler):
    features = features.copy()

    columns_to_scale = ["tenure", "MonthlyCharges"]

    features[columns_to_scale] = scaler.transform(
        features[columns_to_scale]
    )

    return features


def risk_band(probability):
    if probability < 0.30:
        return "Low"
    elif probability < 0.60:
        return "Moderate"
    return "High"


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 📊 ChurnIQ")
    st.caption(tr("Customer Intelligence Platform"))

    st.markdown("---")

    page = st.radio(
        tr("NAVIGATION"),
        ["Overview", "Churn Prediction", "Retention Campaign", "Model Insights", "Metadata"],
        index=0,
        format_func=tr,
    )

    st.markdown("---")

    st.markdown("### " + tr("Model information"))

    st.markdown(
        """
        **Algorithm:** XGBoost

        **Task:** Binary classification

        **Target:** Customer churn
        """
    )

    st.markdown("---")
    st.caption(tr("Built with Python · Scikit-learn · XGBoost · Streamlit"))


# =========================================================
# MODEL STATUS
# =========================================================

if not MODEL_READY:
    st.error(
        "The model files could not be loaded. "
        "Check that the three .pkl files exist inside the 'XGBoost Model' folder."
    )
    st.code(MODEL_ERROR)
    st.stop()


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Customer Analytics · Machine Learning</div>
        <h1>Understand churn.<br>Act before customers leave.</h1>
        <p>
            Explore customer behavior, estimate churn risk, and turn
            machine learning predictions into more informed retention
            decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "Overview":

    section_heading(
        "Project overview",
        "A machine learning solution for customer retention analysis.",
    )

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        metric_card(
            "DATASET SIZE",
            "7,032",
            "Customer records in the source dataset",
        )

    with col2:
        metric_card(
            "MODEL",
            "XGBoost",
            "Tuned binary classification model",
        )

    with col3:
        metric_card(
            "CHURN RECALL",
            "80%",
            "Previously measured on the test set",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        section_heading(
            "The business problem",
            "Why customer churn matters",
        )

        st.markdown(
            """
            Customer churn occurs when customers stop using a service.

            Predicting which customers may leave can help a business
            investigate potential issues and prioritize retention efforts.

            **This project combines:**
            - SQL for structured data analysis.
            - Python and Pandas for data preparation.
            - Exploratory Data Analysis to investigate customer behavior.
            - XGBoost to estimate customer churn.
            - Streamlit to make the model interactive.
            """
        )

    with right:
        section_heading(
            "Prediction workflow",
            "From customer information to model output",
        )

        st.markdown(
            """
            **01 — Customer profile**

            Enter subscription, service, and billing information.

            **02 — Feature preparation**

            Encode the inputs and apply the saved scaler.

            **03 — Model inference**

            Pass the prepared features to XGBoost.

            **04 — Risk interpretation**

            Review the predicted class and estimated probability.
            """
        )

    st.info(
        "The displayed test metrics are historical evaluation results. "
        "They are not a guarantee of performance on new customers."
    )


# =========================================================
# CHURN PREDICTION PAGE
# =========================================================

elif page == "Churn Prediction":

    section_heading(
        "Customer churn prediction",
        "Enter a customer profile to estimate their churn probability.",
    )

    # Restore values before creating widgets. Values are saved by on_change
    # callbacks, so navigating to another page does not reset the profile.
    _churn_widget_state("gender", "Female", ["Female", "Male"])
    _churn_widget_state("senior_citizen", 0, [0, 1])
    _churn_widget_state("tenure", 12)
    _churn_widget_state("partner", "Yes", ["Yes", "No"])
    _churn_widget_state("dependents", "No", ["Yes", "No"])
    _churn_widget_state("phone_service", "Yes", ["Yes", "No"])
    _churn_widget_state("multiple_lines", "No", ["No", "Yes", "No phone service"])
    _churn_widget_state("internet_service", "DSL", ["DSL", "Fiber optic", "No"])
    for field in ["online_security", "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies"]:
        _churn_widget_state(field, "No", ["No", "Yes", "No internet service"])
    _churn_widget_state("contract", "Month-to-month", ["Month-to-month", "One year", "Two year"])
    _churn_widget_state("payment_method", "Electronic check", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    _churn_widget_state("paperless_billing", "Yes", ["Yes", "No"])
    _churn_widget_state("monthly_charges", 70.0)

    st.markdown("### 01 · Customer profile")
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox(tr("Gender"), ["Female", "Male"], key="cp_widget_gender", on_change=_save_churn_input, args=("gender",))
    with c2:
        senior_citizen = st.selectbox(tr("Senior citizen"), [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="cp_widget_senior_citizen", on_change=_save_churn_input, args=("senior_citizen",))
    with c3:
        tenure = st.slider(tr("Tenure (months)"), min_value=0, max_value=72, key="cp_widget_tenure", on_change=_save_churn_input, args=("tenure",))

    c1, c2 = st.columns(2)
    with c1:
        partner = st.selectbox(tr("Has a partner?"), ["Yes", "No"], key="cp_widget_partner", on_change=_save_churn_input, args=("partner",))
    with c2:
        dependents = st.selectbox(tr("Has dependents?"), ["Yes", "No"], key="cp_widget_dependents", on_change=_save_churn_input, args=("dependents",))

    st.markdown("---")
    st.markdown("### 02 · Subscription & services")
    c1, c2, c3 = st.columns(3)
    with c1:
        phone_service = st.selectbox(tr("Phone service"), ["Yes", "No"], key="cp_widget_phone_service", on_change=_save_churn_input, args=("phone_service",))
    with c2:
        if phone_service == "Yes":
            multiple_lines = st.selectbox(tr("Multiple lines"), ["No", "Yes"], key="cp_widget_multiple_lines", on_change=_save_churn_input, args=("multiple_lines",))
        else:
            multiple_lines = "No phone service"
            st.selectbox(tr("Multiple lines"), ["No phone service"], disabled=True, key="cp_widget_multiple_lines_disabled")
            st.session_state["cp_saved_multiple_lines"] = multiple_lines
    with c3:
        internet_service = st.selectbox(tr("Internet service"), ["DSL", "Fiber optic", "No"], key="cp_widget_internet_service", on_change=_save_churn_input, args=("internet_service",))

    internet_options = ["No", "Yes"] if internet_service != "No" else ["No internet service"]
    st.session_state["cp_saved_internet_options"] = internet_options
    service_fields = [
        ("online_security", "Online security"),
        ("online_backup", "Online backup"),
        ("device_protection", "Device protection"),
        ("tech_support", "Tech support"),
        ("streaming_tv", "Streaming TV"),
        ("streaming_movies", "Streaming movies"),
    ]
    service_values = {}
    cols = st.columns(3)
    for idx, (field, label) in enumerate(service_fields):
        with cols[idx % 3]:
            # When internet is unavailable, use the only valid category.
            if internet_service == "No":
                service_values[field] = "No internet service"
                st.selectbox(tr(label), ["No internet service"], disabled=True, key=f"cp_widget_{field}_disabled")
                st.session_state[f"cp_saved_{field}"] = "No internet service"
            else:
                widget_key = f"cp_widget_{field}"
                if st.session_state.get(widget_key) not in ["No", "Yes"]:
                    st.session_state[widget_key] = st.session_state.get(f"cp_saved_{field}", "No")
                    if st.session_state[widget_key] not in ["No", "Yes"]:
                        st.session_state[widget_key] = "No"
                service_values[field] = st.selectbox(tr(label), ["No", "Yes"], key=widget_key, on_change=_save_churn_input, args=(field,))

    st.markdown("---")
    st.markdown("### 03 · Contract & billing")
    c1, c2, c3 = st.columns(3)
    with c1:
        contract = st.selectbox(tr("Contract type"), ["Month-to-month", "One year", "Two year"], key="cp_widget_contract", on_change=_save_churn_input, args=("contract",))
    with c2:
        payment_method = st.selectbox(tr("Payment method"), ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], key="cp_widget_payment_method", on_change=_save_churn_input, args=("payment_method",))
    with c3:
        paperless_billing = st.selectbox(tr("Paperless billing"), ["Yes", "No"], key="cp_widget_paperless_billing", on_change=_save_churn_input, args=("paperless_billing",))
    c1, c2 = st.columns(2)
    with c1:
        monthly_charges = st.number_input(tr("Monthly charges ($)"), min_value=0.0, max_value=1000.0, step=5.0, key="cp_widget_monthly_charges", on_change=_save_churn_input, args=("monthly_charges",))
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.button("✨ Predict customer churn", use_container_width=True)

    if submitted:
        customer_data = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": service_values["online_security"],
            "OnlineBackup": service_values["online_backup"],
            "DeviceProtection": service_values["device_protection"],
            "TechSupport": service_values["tech_support"],
            "StreamingTV": service_values["streaming_tv"],
            "StreamingMovies": service_values["streaming_movies"],
            "Contract": contract,
            "PaymentMethod": payment_method,
            "PaperlessBilling": paperless_billing,
            "MonthlyCharges": monthly_charges,
        }
        try:
            raw_features = build_customer_features(customer_data, feature_columns)
            prepared_features = scale_customer_features(raw_features, scaler)
            prediction = int(model.predict(prepared_features)[0])
            probability = float(model.predict_proba(prepared_features)[0][1]) if hasattr(model, "predict_proba") else None
            st.markdown("---")
            section_heading("Prediction result", "Model output for the entered customer profile.")
            if prediction == 1:
                st.markdown("""<div class="result-card"><div class="result-risk">⚠ Higher churn concern</div><p class="result-caption">The model predicts that this customer belongs to the churn class. Consider reviewing their experience and retention options.</p></div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="result-card"><div class="result-safe">✓ Lower immediate concern</div><p class="result-caption">The model predicts that this customer belongs to the non-churn class. This does not guarantee that the customer will remain subscribed.</p></div>""", unsafe_allow_html=True)
            if probability is not None:
                probability_percent = probability * 100
                result_col, chart_col = st.columns([0.9, 1.1], gap="large")
                with result_col:
                    metric_card("ESTIMATED CHURN PROBABILITY", f"{probability_percent:.1f}%", "Model-estimated probability for this profile")
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f"**Probability band:** {risk_band(probability)}")
                    st.caption("The bands are illustrative thresholds, not business-validated risk categories.")
                with chart_col:
                    gauge = go.Figure(go.Indicator(mode="gauge+number", value=probability_percent, number={"suffix": "%"}, title={"text": "Churn probability", "font": {"size": 17}}, gauge={"axis": {"range": [0, 100], "ticksuffix": "%"}, "bar": {"color": "#38BDF8"}, "bgcolor": "#111B2E", "borderwidth": 1}))
                    gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", font_color="#EDEFF3")
                    st.plotly_chart(gauge, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not generate prediction: {exc}")


# =========================================================
# MODEL INSIGHTS PAGE
# =========================================================

elif page == "Model Insights":

    section_heading(
        "Model insights",
        "Current evaluation results from the held-out test set.",
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("ACCURACY", "73%", "Overall correct predictions")

    with c2:
        metric_card("CHURN PRECISION", "49%", "Precision for class 1")

    with c3:
        metric_card("CHURN RECALL", "80%", "Recall for class 1")

    with c4:
        metric_card("CHURN F1-SCORE", "61%", "F1-score for class 1")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        ### What do these metrics mean?

        - **Precision:** Of the customers predicted to churn, how many
          actually churned in the test set?
        - **Recall:** Of the customers who actually churned, how many
          did the model identify?
        - **F1-score:** A combined measure of precision and recall.
        - **Accuracy:** The proportion of all test predictions that
          were correct.

        The model achieved relatively high churn recall in the recorded
        evaluation, while precision was lower. That trade-off should be
        evaluated against the business cost of unnecessary retention
        outreach versus missed churn cases.
        """
    )

    st.warning(
        "These are the previously reported test results. Reconfirm them "
        "after finalizing the preprocessing pipeline before publishing "
        "the application."
    )


# =========================================================
# RETENTION CAMPAIGN / BATCH CUSTOMER PRIORITIZATION
# =========================================================

elif page == "Retention Campaign":

    section_heading(
        "Retention campaign prioritization",
        "Enter any number of customer profiles one by one, then prioritize outreach by estimated churn probability.",
    )

    st.info(
        "These thresholds are configurable business rules for prioritizing outreach, "
        "not validated guarantees. The percentage shown is the model-estimated churn probability."
    )

    if "retention_customers" not in st.session_state:
        st.session_state.retention_customers = []
    if "retention_form_version" not in st.session_state:
        st.session_state.retention_form_version = 0

    # Streamlit may discard widget state when navigating to another page.
    # Keep the chosen target in a separate session-state key and restore it.
    if "retention_target_count_saved" not in st.session_state:
        st.session_state.retention_target_count_saved = 10
    if "retention_target_count_widget" not in st.session_state:
        st.session_state.retention_target_count_widget = st.session_state.retention_target_count_saved

    target_count = st.number_input(
        "How many customers do you want to evaluate?",
        min_value=1,
        max_value=500,
        step=1,
        key="retention_target_count_widget",
        on_change=_save_retention_target_count,
        help="The number is flexible: choose any number from 1 to 500.",
    )

    current_count = len(st.session_state.retention_customers)
    st.progress(min(current_count / int(target_count), 1.0))
    st.caption(f"Profiles evaluated: {current_count} / {int(target_count)}")

    if current_count < int(target_count):
        version = st.session_state.retention_form_version
        form_key = f"retention_customer_form_{version}"
        with st.form(form_key):
            st.markdown(f"### Customer {current_count + 1} of {int(target_count)}")
            customer_name = st.text_input(
                tr("Customer name or reference ID"),
                value=f"Customer {current_count + 1}",
                key=f"retention_name_{version}",
            )

            st.markdown("#### 01 · Customer profile")
            c1, c2, c3 = st.columns(3)
            with c1:
                gender = st.selectbox(tr("Gender"), ["Female", "Male"], key=f"ret_gender_{version}")
            with c2:
                senior_citizen = st.selectbox(tr("Senior citizen"), [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key=f"ret_senior_{version}")
            with c3:
                tenure = st.slider(tr("Tenure (months)"), 0, 72, 12, key=f"ret_tenure_{version}")

            c1, c2 = st.columns(2)
            with c1:
                partner = st.selectbox(tr("Has a partner?"), ["Yes", "No"], key=f"ret_partner_{version}")
            with c2:
                dependents = st.selectbox(tr("Has dependents?"), ["Yes", "No"], key=f"ret_dependents_{version}")

            st.markdown("#### 02 · Subscription & services")
            c1, c2, c3 = st.columns(3)
            with c1:
                phone_service = st.selectbox(tr("Phone service"), ["Yes", "No"], key=f"ret_phone_{version}")
            with c2:
                if phone_service == "Yes":
                    multiple_lines = st.selectbox(tr("Multiple lines"), ["No", "Yes"], key=f"ret_lines_{version}")
                else:
                    multiple_lines = "No phone service"
                    st.selectbox(tr("Multiple lines"), ["No phone service"], disabled=True, key=f"ret_lines_{version}")
            with c3:
                internet_service = st.selectbox(tr("Internet service"), ["DSL", "Fiber optic", "No"], key=f"ret_internet_{version}")

            internet_options = ["No", "Yes"] if internet_service != "No" else ["No internet service"]
            c1, c2, c3 = st.columns(3)
            with c1:
                online_security = st.selectbox(tr("Online security"), internet_options, disabled=internet_service == "No", key=f"ret_security_{version}")
            with c2:
                online_backup = st.selectbox(tr("Online backup"), internet_options, disabled=internet_service == "No", key=f"ret_backup_{version}")
            with c3:
                device_protection = st.selectbox(tr("Device protection"), internet_options, disabled=internet_service == "No", key=f"ret_device_{version}")

            c1, c2, c3 = st.columns(3)
            with c1:
                tech_support = st.selectbox(tr("Tech support"), internet_options, disabled=internet_service == "No", key=f"ret_support_{version}")
            with c2:
                streaming_tv = st.selectbox(tr("Streaming TV"), internet_options, disabled=internet_service == "No", key=f"ret_tv_{version}")
            with c3:
                streaming_movies = st.selectbox(tr("Streaming movies"), internet_options, disabled=internet_service == "No", key=f"ret_movies_{version}")

            st.markdown("#### 03 · Contract & billing")
            c1, c2, c3 = st.columns(3)
            with c1:
                contract = st.selectbox(tr("Contract type"), ["Month-to-month", "One year", "Two year"], key=f"ret_contract_{version}")
            with c2:
                payment_method = st.selectbox(tr("Payment method"), ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], key=f"ret_payment_{version}")
            with c3:
                paperless_billing = st.selectbox(tr("Paperless billing"), ["Yes", "No"], key=f"ret_paperless_{version}")
            monthly_charges = st.number_input(tr("Monthly charges ($)"), min_value=0.0, max_value=1000.0, value=70.0, step=5.0, key=f"ret_charges_{version}")

            add_customer = st.form_submit_button(
                tr("Add customer and calculate churn risk"),
                use_container_width=True,
                type="primary",
            )

        if add_customer:
            customer_data = {
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
                "PaymentMethod": payment_method,
                "PaperlessBilling": paperless_billing,
                "MonthlyCharges": monthly_charges,
            }
            try:
                raw_features = build_customer_features(customer_data, feature_columns)
                prepared_features = scale_customer_features(raw_features, scaler)
                prediction = int(model.predict(prepared_features)[0])
                probability = float(model.predict_proba(prepared_features)[0][1]) if hasattr(model, "predict_proba") else None
                if probability is None:
                    st.error("The loaded model does not provide predict_proba, so customers cannot be ranked by probability.")
                else:
                    st.session_state.retention_customers.append({
                        tr("Customer"): customer_name.strip() or f"Customer {current_count + 1}",
                        tr("Churn probability (%)"): round(probability * 100, 1),
                        tr("Prediction"): "Churn" if prediction == 1 else "No churn",
                        tr("Priority"): (
                            "Urgent — above 66%" if probability > 0.66 else
                            "Recommended — 45–66%" if probability >= 0.45 else
                            "Lower priority — 33–<45%" if probability >= 0.33 else
                            "Optional — below 33%"
                        ),
                        "_probability": probability,
                        "_customer_data": customer_data.copy(),
                    })
                    st.session_state.retention_form_version += 1
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not score this customer: {exc}")

    customers = st.session_state.retention_customers
    if customers:
        st.markdown("---")
        st.markdown("### Current customer list")
        results = pd.DataFrame(customers).sort_values("_probability", ascending=False).reset_index(drop=True)
        display_results = results.drop(columns=["_probability"])
        st.dataframe(display_results, use_container_width=True, hide_index=True)

        if len(customers) >= int(target_count):
            st.success(f"All {len(customers)} customer profiles have been evaluated. The list below is sorted from highest to lowest estimated churn probability.")
        else:
            st.caption("Continue adding customer profiles above. Results are ranked as each profile is submitted.")

        # Edit an already-added customer. The raw profile is stored separately
        # from the display table so edits can be rescored with the same model.
        st.markdown("### Edit a customer profile")
        editable_indices = [i for i, item in enumerate(st.session_state.retention_customers)
                            if item.get("_customer_data") is not None]
        if editable_indices:
            edit_labels = {
                i: f'{st.session_state.retention_customers[i].get(tr("Customer"), f"Customer {i+1}")} '
                   f'— {st.session_state.retention_customers[i].get(tr("Churn probability (%)"), "?")}%'
                for i in editable_indices
            }
            selected_index = st.selectbox(
                "Select customer to edit",
                options=editable_indices,
                format_func=lambda i: edit_labels[i],
                key="retention_edit_selected_index",
            )
            selected_customer = st.session_state.retention_customers[selected_index]
            original = selected_customer["_customer_data"].copy()
            edit_name = st.text_input(
                "Customer name or reference ID",
                value=selected_customer.get(tr("Customer"), f"Customer {selected_index+1}"),
                key=f"retention_edit_name_{selected_index}",
            )
            with st.form(f"retention_edit_form_{selected_index}"):
                st.caption("Update the fields below, then save to recalculate this customer's churn risk.")
                ec1, ec2, ec3 = st.columns(3)
                with ec1:
                    e_gender = st.selectbox("Gender", ["Female", "Male"], index=["Female", "Male"].index(original.get("gender", "Female")), key=f"edit_gender_{selected_index}")
                with ec2:
                    e_senior = st.selectbox("Senior citizen", [0, 1], index=int(original.get("SeniorCitizen", 0)), format_func=lambda x: "Yes" if x else "No", key=f"edit_senior_{selected_index}")
                with ec3:
                    e_tenure = st.slider("Tenure (months)", 0, 72, int(original.get("tenure", 12)), key=f"edit_tenure_{selected_index}")
                ec1, ec2 = st.columns(2)
                with ec1:
                    e_partner = st.selectbox("Has a partner?", ["Yes", "No"], index=["Yes", "No"].index(original.get("Partner", "No")), key=f"edit_partner_{selected_index}")
                with ec2:
                    e_dependents = st.selectbox("Has dependents?", ["Yes", "No"], index=["Yes", "No"].index(original.get("Dependents", "No")), key=f"edit_dependents_{selected_index}")
                st.markdown("#### Services")
                ec1, ec2, ec3 = st.columns(3)
                with ec1:
                    e_phone = st.selectbox("Phone service", ["Yes", "No"], index=["Yes", "No"].index(original.get("PhoneService", "Yes")), key=f"edit_phone_{selected_index}")
                with ec2:
                    line_opts = ["No phone service"] if e_phone == "No" else ["No", "Yes"]
                    old_line = original.get("MultipleLines", "No")
                    e_lines = st.selectbox("Multiple lines", line_opts, index=line_opts.index(old_line) if old_line in line_opts else 0, key=f"edit_lines_{selected_index}")
                with ec3:
                    internet_opts = ["DSL", "Fiber optic", "No"]
                    e_internet = st.selectbox("Internet service", internet_opts, index=internet_opts.index(original.get("InternetService", "DSL")), key=f"edit_internet_{selected_index}")
                service_opts = ["No internet service"] if e_internet == "No" else ["No", "Yes"]
                service_cols = st.columns(3)
                service_fields = [("Online security", "OnlineSecurity"), ("Online backup", "OnlineBackup"), ("Device protection", "DeviceProtection"), ("Tech support", "TechSupport"), ("Streaming TV", "StreamingTV"), ("Streaming movies", "StreamingMovies")]
                edited_services = {}
                for j, (label, field) in enumerate(service_fields):
                    with service_cols[j % 3]:
                        old_value = original.get(field, "No")
                        idx = service_opts.index(old_value) if old_value in service_opts else 0
                        edited_services[field] = st.selectbox(label, service_opts, index=idx, key=f"edit_{field}_{selected_index}", disabled=e_internet == "No")
                st.markdown("#### Contract & billing")
                ec1, ec2, ec3 = st.columns(3)
                contract_opts = ["Month-to-month", "One year", "Two year"]
                payment_opts = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
                with ec1:
                    e_contract = st.selectbox("Contract type", contract_opts, index=contract_opts.index(original.get("Contract", "Month-to-month")), key=f"edit_contract_{selected_index}")
                with ec2:
                    e_payment = st.selectbox("Payment method", payment_opts, index=payment_opts.index(original.get("PaymentMethod", "Electronic check")), key=f"edit_payment_{selected_index}")
                with ec3:
                    e_paperless = st.selectbox("Paperless billing", ["Yes", "No"], index=["Yes", "No"].index(original.get("PaperlessBilling", "Yes")), key=f"edit_paperless_{selected_index}")
                e_charges = st.number_input("Monthly charges ($)", min_value=0.0, max_value=1000.0, value=float(original.get("MonthlyCharges", 70.0)), step=5.0, key=f"edit_charges_{selected_index}")
                save_edit = st.form_submit_button("Save changes and recalculate", type="primary", use_container_width=True)
            if save_edit:
                updated_data = {
                    "gender": e_gender, "SeniorCitizen": e_senior, "Partner": e_partner,
                    "Dependents": e_dependents, "tenure": e_tenure, "PhoneService": e_phone,
                    "MultipleLines": e_lines, "InternetService": e_internet,
                    **edited_services, "Contract": e_contract, "PaymentMethod": e_payment,
                    "PaperlessBilling": e_paperless, "MonthlyCharges": e_charges,
                }
                if e_internet == "No":
                    for field in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
                        updated_data[field] = "No internet service"
                try:
                    updated_features = build_customer_features(updated_data, feature_columns)
                    updated_features = scale_customer_features(updated_features, scaler)
                    updated_probability = float(model.predict_proba(updated_features)[0][1])
                    updated_prediction = int(model.predict(updated_features)[0])
                    selected_customer[tr("Customer")] = edit_name.strip() or f"Customer {selected_index+1}"
                    selected_customer[tr("Churn probability (%)")] = round(updated_probability * 100, 1)
                    selected_customer[tr("Prediction")] = "Churn" if updated_prediction == 1 else "No churn"
                    selected_customer[tr("Priority")] = (
                        "Urgent — above 66%" if updated_probability > 0.66 else
                        "Recommended — 45–66%" if updated_probability >= 0.45 else
                        "Lower priority — 33–<45%" if updated_probability >= 0.33 else
                        "Optional — below 33%"
                    )
                    selected_customer["_probability"] = updated_probability
                    selected_customer["_customer_data"] = updated_data.copy()
                    st.session_state.retention_customers[selected_index] = selected_customer
                    st.success("Customer updated and risk recalculated.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not update this customer: {exc}")
        else:
            st.caption("No editable customer profiles are available yet. Add a customer above first.")

        st.markdown("### Outreach priority groups")
        priority_groups = [
            ("🚨 Urgent outreach", "Above 66% — prioritize these customers for retention offers.", lambda p: p > 0.66),
            ("🟠 Recommended outreach", "45% to 66% inclusive — outreach is recommended.", lambda p: 0.45 <= p <= 0.66),
            ("🟡 Lower-priority outreach", "33% to below 45% — consider outreach if budget allows.", lambda p: 0.33 <= p < 0.45),
            ("🟢 Optional outreach", "Below 33% — no immediate outreach is suggested by these rules.", lambda p: p < 0.33),
        ]
        for title, description, predicate in priority_groups:
            group = results[results["_probability"].map(predicate)].sort_values("_probability", ascending=False)
            with st.expander(f"{title} · {len(group)} customer(s)", expanded=(title.startswith("🚨"))):
                st.caption(description)
                if group.empty:
                    st.write("No customers in this group yet.")
                else:
                    st.dataframe(group.drop(columns=["_probability"]), use_container_width=True, hide_index=True)

        csv = results.drop(columns=["_probability", "_customer_data"], errors="ignore").to_csv(index=False).encode("utf-8-sig")
        st.download_button(tr("Download prioritized customer list (CSV)"), data=csv, file_name="churn_retention_priorities.csv", mime="text/csv")
        if st.button(tr("Clear customer list and start a new campaign"), use_container_width=True):
            st.session_state.retention_customers = []
            st.session_state.retention_form_version += 1
            st.rerun()


# =========================================================
# METADATA / DATA DICTIONARY PAGE
# =========================================================

elif page == "Metadata":

    section_heading(
        tr("Dataset metadata"),
        tr("A data dictionary explaining what each field represents and how to interpret its categories."),
    )

    st.info(tr("The dataset contains customer demographics, subscribed services, contract and billing details, and the churn outcome. The descriptions below explain the original dataset fields; not every field is used as an input to the current XGBoost model."))

    metadata_rows = [
        {"Category": "Customer profile", "Field": "gender", "Meaning": "Customer's recorded gender.", "Values / interpretation": "Female · Male", "Used by model": "Yes"},
        {"Category": "Customer profile", "Field": "SeniorCitizen", "Meaning": "Whether the customer is classified as a senior citizen.", "Values / interpretation": "0 = No · 1 = Yes", "Used by model": "Yes"},
        {"Category": "Customer profile", "Field": "Partner", "Meaning": "Whether the customer has a partner.", "Values / interpretation": "Yes · No", "Used by model": "Yes"},
        {"Category": "Customer profile", "Field": "Dependents", "Meaning": "Whether the customer has dependents.", "Values / interpretation": "Yes · No", "Used by model": "Yes"},
        {"Category": "Account", "Field": "customerID", "Meaning": "Unique identifier assigned to each customer; it does not describe customer behavior.", "Values / interpretation": "Text ID, e.g. 7590-VHVEG", "Used by model": "No — identifier only"},
        {"Category": "Account", "Field": "tenure", "Meaning": "How long the customer has stayed with the company.", "Values / interpretation": "Number of months (0–72 in the app)", "Used by model": "Yes"},
        {"Category": "Phone services", "Field": "PhoneService", "Meaning": "Whether the customer subscribes to phone service.", "Values / interpretation": "Yes · No", "Used by model": "Yes"},
        {"Category": "Phone services", "Field": "MultipleLines", "Meaning": "Whether the customer has multiple phone lines.", "Values / interpretation": "Yes · No · No phone service", "Used by model": "Yes"},
        {"Category": "Internet services", "Field": "InternetService", "Meaning": "Type of internet connection subscribed to.", "Values / interpretation": "DSL · Fiber optic · No", "Used by model": "Yes"},
        {"Category": "Internet services", "Field": "OnlineSecurity", "Meaning": "Whether online security service is included.", "Values / interpretation": "Yes · No · No internet service", "Used by model": "Yes"},
        {"Category": "Internet services", "Field": "OnlineBackup", "Meaning": "Whether online backup service is included.", "Values / interpretation": "Yes · No · No internet service", "Used by model": "Yes"},
        {"Category": "Internet services", "Field": "DeviceProtection", "Meaning": "Whether device protection service is included.", "Values / interpretation": "Yes · No · No internet service", "Used by model": "Yes"},
        {"Category": "Internet services", "Field": "TechSupport", "Meaning": "Whether technical support service is included.", "Values / interpretation": "Yes · No · No internet service", "Used by model": "Yes"},
        {"Category": "Internet services", "Field": "StreamingTV", "Meaning": "Whether the customer has the TV streaming add-on.", "Values / interpretation": "Yes · No · No internet service", "Used by model": "Yes"},
        {"Category": "Internet services", "Field": "StreamingMovies", "Meaning": "Whether the customer has the movie streaming add-on.", "Values / interpretation": "Yes · No · No internet service", "Used by model": "Yes"},
        {"Category": "Contract & billing", "Field": "Contract", "Meaning": "Length/type of the customer's service agreement.", "Values / interpretation": "Month-to-month · One year · Two year", "Used by model": "Yes"},
        {"Category": "Contract & billing", "Field": "PaperlessBilling", "Meaning": "Whether the customer receives billing electronically without paper statements.", "Values / interpretation": "Yes · No", "Used by model": "Yes"},
        {"Category": "Contract & billing", "Field": "PaymentMethod", "Meaning": "Payment method used for the service bill.", "Values / interpretation": "Electronic check · Mailed check · Bank transfer (automatic) · Credit card (automatic)", "Used by model": "Yes"},
        {"Category": "Contract & billing", "Field": "MonthlyCharges", "Meaning": "Amount charged to the customer each month.", "Values / interpretation": "Numeric amount in dollars ($)", "Used by model": "Yes"},
        {"Category": "Contract & billing", "Field": "TotalCharges", "Meaning": "Total amount charged to the customer over the recorded relationship.", "Values / interpretation": "Numeric amount in dollars ($)", "Used by model": "No — not in current model features"},
        {"Category": "Churn outcome", "Field": "Churn", "Meaning": "Whether the customer left the company; this is the outcome the model is designed to predict.", "Values / interpretation": "Yes = churned · No = stayed", "Used by model": "Target label, not an input"},
    ]

    metadata_df = pd.DataFrame(metadata_rows)
    # Translate the display table only; original field names and model values remain unchanged.
    metadata_display = metadata_df.copy()
    for column in ["Category", "Meaning", "Values / interpretation", "Used by model"]:
        metadata_display[column] = metadata_display[column].map(tr)
    category_names = ["All categories"] + list(dict.fromkeys(metadata_df["Category"].tolist()))
    selected_category = st.selectbox("Filter by category", category_names, format_func=tr)

    if selected_category != "All categories":
        visible_df = metadata_display[metadata_df["Category"] == selected_category]
    else:
        visible_df = metadata_display

    st.dataframe(
        visible_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Category": st.column_config.TextColumn(tr("Category"), width="medium"),
            "Field": st.column_config.TextColumn(tr("Field"), width="medium"),
            "Meaning": st.column_config.TextColumn(tr("What it means"), width="large"),
            "Values / interpretation": st.column_config.TextColumn(tr("Possible values"), width="large"),
            "Used by model": st.column_config.TextColumn(tr("Role in model"), width="medium"),
        },
    )

    st.markdown("### " + tr("How to read the categories"))

    st.markdown(
        """
        - **Yes / No:** indicates whether the customer subscribes to a service or has a given attribute.
        - **No phone service:** the customer does not have phone service, so multiple-line status does not apply.
        - **No internet service:** the customer does not subscribe to internet service, so internet add-ons do not apply.
        - **Contract type:** month-to-month contracts renew monthly; one-year and two-year contracts have longer terms.
        - **Churn:** the historical outcome used as the prediction target. When predicting a new customer, this value is not supplied as an input.
        - **Role in model:** identifies whether a field is an input feature, an identifier, or the target label. `TotalCharges` is present in the dataset but is not one of the current 22 model features.
        """
    )

st.markdown("---")

st.caption(tr("ChurnIQ · Telco Customer Churn Prediction · Educational portfolio project"))