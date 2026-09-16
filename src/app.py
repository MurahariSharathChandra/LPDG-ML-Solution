import streamlit as st
from pathlib import Path
import pandas as pd
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = ROOT / "reports" / "final_model.joblib"
FEATURE_FILE = ROOT / "reports" / "final_features.txt"

THRESHOLD = 0.45


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="LPDG Fault Prediction",
    page_icon="⚡",
    layout="wide",
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_features():
    with open(FEATURE_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


model = load_model()
features = load_features()


# ============================================================
# HEADER
# ============================================================

st.title("⚡ LPDG Gateway Fault Prediction")
st.subheader("Predictive Maintenance & Fault Risk Assessment")

st.write(
    "This system uses historical gateway telemetry to estimate "
    "the probability of an operational fault."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Model Information")

st.sidebar.metric(
    "Model Features",
    len(features),
)

st.sidebar.metric(
    "Decision Threshold",
    f"{THRESHOLD:.2f}",
)

st.sidebar.info(
    "The final Random Forest model was validated using "
    "5-fold cross-validation."
)


# ============================================================
# INPUT SECTION
# ============================================================

st.header("Gateway Telemetry Input")

st.write(
    "Enter telemetry values below. "
    "Values not provided will be treated as missing."
)


# Create two columns
col1, col2 = st.columns(2)


input_values = {}


# ------------------------------------------------------------
# Important features shown to user
# ------------------------------------------------------------

important_features = [
    "offline_duration_sec_r7d",
    "offline_duration_sec_r7d_sum",
    "read_rate",
    "offline_duration_sec_r28d",
    "offline_duration_sec_r28d_sum",
    "read_rate_4w",
    "online_duration_mins_r7d",
    "rsrp_bad_share_r28d",
    "disconnection_cnt_r7d_sum",
    "reboot_duration_sec_r7d",
]


with col1:

    st.subheader("Connectivity")

    for feature in important_features[:5]:

        if feature in features:

            input_values[feature] = st.number_input(
                feature,
                value=0.0,
                step=0.1,
            )


with col2:

    st.subheader("Network & Activity")

    for feature in important_features[5:]:

        if feature in features:

            input_values[feature] = st.number_input(
                feature,
                value=0.0,
                step=0.1,
            )


# ============================================================
# PREDICTION
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Fault Risk",
    type="primary",
    use_container_width=True,
)


if predict_button:

    # --------------------------------------------------------
    # Create dataframe with ALL model features
    # --------------------------------------------------------

    input_data = {}

    for feature in features:

        if feature in input_values:
            input_data[feature] = input_values[feature]

        else:
            input_data[feature] = None

    input_df = pd.DataFrame(
        [input_data],
        columns=features,
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probability = model.predict_proba(
        input_df
    )[0][1]

    prediction = int(
        probability >= THRESHOLD
    )


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    st.header("Prediction Result")

    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "Fault Probability",
            f"{probability * 100:.2f}%",
        )


    with result_col2:

        if prediction == 1:

            st.error(
                "⚠️ FAULT LIKELY"
            )

        else:

            st.success(
                "✅ NO FAULT"
            )


    # --------------------------------------------------------
    # Risk interpretation
    # --------------------------------------------------------

    st.subheader("Risk Interpretation")

    if probability >= 0.75:

        st.error(
            "High risk: the telemetry pattern indicates "
            "a strong likelihood of an operational fault."
        )

    elif probability >= THRESHOLD:

        st.warning(
            "Moderate risk: the gateway shows telemetry "
            "patterns associated with potential faults."
        )

    else:

        st.success(
            "Lower risk: the current telemetry pattern "
            "does not cross the fault decision threshold."
        )


    # --------------------------------------------------------
    # Probability bar
    # --------------------------------------------------------

    st.progress(
        min(probability, 1.0)
    )


    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    with st.expander("Model Details"):

        st.write(
            f"Number of model features: {len(features)}"
        )

        st.write(
            f"Decision threshold: {THRESHOLD}"
        )

        st.write(
            "Prediction generated using the trained "
            "Random Forest model."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "LPDG Predictive Maintenance System | "
    "Machine Learning Fault Prediction"
)