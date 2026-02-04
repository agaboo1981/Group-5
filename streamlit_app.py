import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Risk Predictor | Civil Eng Group 5",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Model Loading (Non-Blocking)
# -------------------------
model = None
model_path = "final_risk_model.pkl"

try:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        st.error(f"⚠️ Model file '{model_path}' not found. Prediction will be disabled.")
except Exception as e:
    st.error(f"⚠️ Error loading model: {str(e)}")

# -------------------------
# Custom Styling
# -------------------------
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">
<style>
[data-testid="stNumberInput"] button {
    display: none !important;
}
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #dee2e6;
}
section[data-testid="stSidebar"] * {
    color: #2c3e50 !important;
}
.main {
    background-color: #f8f9fa;
    color: #212529 !important;
}
.stButton>button {
    width: 100%;
    background-color: #007bff;
    color: white;
    font-size: 18px;
    padding: 10px;
    border-radius: 10px;
    border: none;
}
.stButton>button:hover {
    background-color: #0056b3;
}
h1, h2, h3 {
    color: #2c3e50 !important;
}
.info-box {
    background-color:#010008;
    color: white !important;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #3498db;
    margin-bottom: 20px;
}
.custom-icon {
    color: #007bff;
    margin-right: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    if os.path.exists("group5.png"):
        st.image("group5.png", width=100)
    st.title("Civil Engineering Group 5")
    st.markdown("---")
    st.info(
        """
        **Project:** Building Collapse Risk Prediction
        
        **Group Members:**
        - Adesina Joshua
        - Adesina Opemipo
        - Adeyemi Abeeb
        - Afolabi Ibrahim
        - Ajayi Moruf
        - Ajayi Serah
        """
    )

# -------------------------
# Main Header
# -------------------------
st.markdown("<h1><i class='fas fa-hard-hat custom-icon'></i> Construction Project Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    Welcome to the Risk Assessment Tool. Please input the project details below to evaluate the potential risk level.
</div>
""", unsafe_allow_html=True)

# -------------------------
# Input Form
# -------------------------
st.markdown("<h3><i class='fas fa-folder-open custom-icon'></i> Project Information</h3>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    project_type = st.selectbox("Project Type", ["Tunnel", "Dam", "Building", "Road"])
with col2:
    location = st.selectbox("Location", ["Lagos", "Kano", "Rivers", "Abuja", "Kaduna"])
with col3:
    weather = st.selectbox("Weather Condition", ["Sunny", "Cloudy", "Harmattan"])
with col4:
    anomaly = st.selectbox("Anomaly Detected?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

st.markdown("---")
st.markdown("<h3><i class='fas fa-coins custom-icon'></i> Financials & Timeline (Key Drivers)</h3>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

with c1:
    planned_cost = st.number_input("Planned Cost (₦)", min_value=1.0, value=1000000.0, step=1000.0)
with c2:
    actual_cost = st.number_input("Actual Cost (₦)", min_value=0.0, value=950000.0, step=1000.0)
with c3:
    planned_duration = st.number_input("Planned Duration (Days)", min_value=1.0, value=100.0, step=1.0)
with c4:
    actual_duration = st.number_input("Actual Duration (Days)", min_value=0.0, value=95.0, step=1.0)

st.markdown("---")
st.markdown("<h3><i class='fas fa-cogs custom-icon'></i> Structural & Environmental Metrics</h3>", unsafe_allow_html=True)
m1, m2, m3 = st.columns(3)

with m1:
    vibration = st.number_input("Vibration Level (m/s²)", min_value=0.0, value=0.05, format="%.4f")
    crack_width = st.number_input("Crack Width (mm)", min_value=0.0, value=0.1, format="%.4f")
    load_capacity = st.number_input("Load Bearing Capacity (MPa)", min_value=1.0, value=30.0)

with m2:
    temperature = st.slider("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
    humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    air_quality = st.slider("Air Quality Index", min_value=0, max_value=500, value=100)

with m3:
    energy = st.number_input("Energy Consumption (kWh)", min_value=0.0, value=500.0)
    material = st.number_input("Material Usage (kg)", min_value=0.0, value=1000.0)

st.markdown("---")
st.markdown("<h3><i class='fas fa-users-cog custom-icon'></i> Resources & Safety</h3>", unsafe_allow_html=True)
r1, r2, r3 = st.columns(3)

labor_calculated = actual_duration * 8.0

with r1:
    labor = st.number_input("Labor Hours (Auto-calculated)", value=labor_calculated, disabled=True)
with r2:
    equipment = st.slider("Equipment Utilization (%)", min_value=0.0, max_value=100.0, value=80.0)

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("Analyze Risk Level")

# -------------------------
# Prediction Logic
# -------------------------
if predict_btn:
    if model is None:
        st.error("Cannot predict: Model file is missing or failed to load.")
    else:
        with st.spinner("Analyzing project parameters..."):
            # 1. Feature Engineering (match model_train.py logic)
            cost_ratio = actual_cost / (planned_cost + 1)
            duration_ratio = actual_duration / (planned_duration + 1)
            load_temp_ratio = load_capacity / (temperature + 273.15)

            # 2. Create Dictionary with correct feature names
            # Note: Removed "(Naira)", "(Days)" etc from keys to match typical CSV output
            input_data = {
                "Project_Type": project_type,
                "Location": location,
                "Weather_Condition": weather,
                "Anomaly_Detected": anomaly,
                "Cost_Ratio": cost_ratio,
                "Duration_Ratio": duration_ratio,
                "Load_Temp_Ratio": load_temp_ratio,
                "Actual_Cost": actual_cost,  # Kept if model uses it, otherwise get_dummies handles it
                "Actual_Duration": actual_duration,
                "Vibration_Level": vibration,
                "Crack_Width": crack_width,
                "Load_Bearing_Capacity": load_capacity,
                "Temperature": temperature,
                "Humidity": humidity,
                "Air_Quality_Index": air_quality,
                "Energy_Consumption": energy,
                "Material_Usage": material,
                "Labor_Hours": labor,
                "Equipment_Utilization": equipment,
            }
            
            input_df = pd.DataFrame([input_data])

            # 3. One-hot encoding
            input_df = pd.get_dummies(input_df)

            # 4. Align with model features (Add missing columns as 0)
            if hasattr(model, 'feature_names_in_'):
                model_features = model.feature_names_in_
                input_df = input_df.reindex(columns=model_features, fill_value=0)
            
            try:
                # 5. Predict
                prediction = model.predict(input_df)[0]

                st.divider()
                st.markdown("<h3><i class='fas fa-chart-line custom-icon'></i> Analysis Result</h3>", unsafe_allow_html=True)
                
                # Display logic
                if "High" in str(prediction) or "Collapse" in str(prediction):
                    st.markdown(f"""
                    <div style="background-color: #f8d7da; color: #721c24; padding: 20px; border-radius: 10px; border-left: 5px solid #bd2130;">
                        <h4 style="margin: 0;"><i class="fas fa-exclamation-triangle"></i> Prediction: {prediction}</h4>
                        <p style="margin: 10px 0 0 0;">High risk detected. Immediate structural audit recommended.</p>
                    </div>""", unsafe_allow_html=True)
                elif "Medium" in str(prediction):
                    st.markdown(f"""
                    <div style="background-color: #fff3cd; color: #856404; padding: 20px; border-radius: 10px; border-left: 5px solid #ffc107;">
                        <h4 style="margin: 0;"><i class="fas fa-exclamation-circle"></i> Prediction: {prediction}</h4>
                        <p style="margin: 10px 0 0 0;">Moderate risk detected. Monitor load capacity closely.</p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #d4edda; color: #155724; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745;">
                        <h4 style="margin: 0;"><i class="fas fa-check-circle"></i> Prediction: {prediction}</h4>
                        <p style="margin: 10px 0 0 0;">Project parameters indicate stable/low risk.</p>
                    </div>""", unsafe_allow_html=True)

                # Graph
                st.markdown("---")
                st.markdown("### <i class='fas fa-wave-square custom-icon'></i> Structural Analysis Graph", unsafe_allow_html=True)
                
                t = np.linspace(0, 1, 200)
                base_amp = max(vibration, 0.1)
                freq = 4 + (base_amp * 0.2)
                signal = (base_amp * np.sin(2 * np.pi * freq * t)) + np.random.normal(0, base_amp * 0.1, t.shape)
                
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(t, signal, color="#007bff", linewidth=1.5)
                ax.set_ylabel("Amplitude")
                ax.set_title("Vibration Signal Analysis")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.info("Check if inputs match the training data columns.")
