import streamlit as st
import joblib
import pandas as pd
import numpy as np
from preprocessing import PreprocessingPipeline

# ==============================
#        PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Machine Failure Prediction Dashboard",
    page_icon="⚙️",
    layout="centered"
)

# ==============================
#        LOAD MODELS
# ==============================
@st.cache_resource
def load_models():
    preprocessor = joblib.load("preprocessing_pipeline.pkl")
    binary_model = joblib.load("best_model_binary_final.pkl")  # change name if timestamped
    multiclass_model = joblib.load("best_model_multiclass_final.pkl")  # change name if timestamped
    return preprocessor, binary_model, multiclass_model

try:
    preprocessor, binary_model, multiclass_model = load_models()
    failure_le = preprocessor.failure_encoder
except Exception as e:
    st.error(f"❌ Error loading models or preprocessor: {e}")
    st.stop()

# ==============================
#        INPUT FUNCTION
# ==============================
def user_input_features():
    st.sidebar.header("🔧 Input Machine Parameters")

    type_input = st.sidebar.selectbox("Type", options=["L", "M", "H"])
    rotational_speed = st.sidebar.number_input("Rotational Speed (rpm)", min_value=0, max_value=3000, value=1500)
    torque = st.sidebar.number_input("Torque (Nm)", min_value=0.0, max_value=100.0, value=30.0)
    tool_wear = st.sidebar.number_input("Tool Wear (min)", min_value=0.0, max_value=300.0, value=100.0)
    air_temp = st.sidebar.number_input("Air Temperature (°C)", min_value=0.0, max_value=400.0, value=25.0)
    process_temp = st.sidebar.number_input("Process Temperature (°C)", min_value=0.0, max_value=400.0, value=70.0)

    data = {
        "Type": [type_input],
        "Rotational_speed_rpm": [rotational_speed],
        "Torque_Nm": [torque],
        "Tool_wear_min": [tool_wear],
        "Air_temperature_C": [air_temp],
        "Process_temperature_C": [process_temp],
    }

    return pd.DataFrame(data)

# ==============================
#        PREDICTION FUNCTION
# ==============================
def preprocess_and_predict(features_df, predict_multiclass=False):
    features_processed = preprocessor.transform(features_df)

    # Binary prediction
    failure_pred = binary_model.predict(features_processed)[0]
    failure_proba = binary_model.predict_proba(features_processed)[0][1]

    if failure_pred == 0:
        return "No Failure", None, failure_proba

    # If predicted failure and multiclass enabled
    if predict_multiclass:
        failure_type_encoded = multiclass_model.predict(features_processed)[0]
        failure_type = failure_le.inverse_transform([failure_type_encoded])[0]
        failure_type_proba = np.max(multiclass_model.predict_proba(features_processed))
        return "Failure Predicted", (failure_type, failure_type_proba), failure_proba
    else:
        return "Failure Predicted", None, failure_proba

# ==============================
#        MAIN APP
# ==============================
def main():
    st.title("⚙️ Machine Failure Prediction Dashboard")
    st.markdown(
        """
        <style>
        .success-box {background-color:#eafbea; padding:15px; border-radius:10px;}
        .failure-box {background-color:#ffe6e6; padding:15px; border-radius:10px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("Predict machine failures and their types using your trained models.")

    task = st.selectbox("Select Prediction Task", ["Binary Classification", "Multiclass Classification"])
    user_features = user_input_features()

    if st.button("🔍 Predict"):
        with st.spinner("Running prediction..."):
            result, failure_info, prob = preprocess_and_predict(user_features, predict_multiclass=(task == "Multiclass Classification"))

        st.subheader("📊 Prediction Result")

        if result == "No Failure":
            st.markdown(f"<div class='success-box'><b>✅ No Failure Detected</b><br>Confidence: {(1 - prob) * 100:.2f}%</div>", unsafe_allow_html=True)
        else:
            if failure_info:
                fail_type, fail_conf = failure_info
                st.markdown(f"<div class='failure-box'><b>⚠️ Failure Predicted</b><br>Type: <b>{fail_type}</b><br>Confidence: {fail_conf * 100:.2f}%</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='failure-box'><b>⚠️ Failure Predicted</b><br>Confidence: {prob * 100:.2f}%</div>", unsafe_allow_html=True)

    # Show input table
    st.markdown("---")
    st.markdown("### 🧾 Input Summary")
    st.dataframe(user_features, use_container_width=True)


if __name__ == "__main__":
    main()
