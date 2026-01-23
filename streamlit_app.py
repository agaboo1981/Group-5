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

# Change the model path to match what you saved
model_path = "random_forest_risk_model.pkl"

if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
    st.info(f"Current directory: {os.getcwd()}")
    st.info(f"Files in directory: {os.listdir('.')}")
else:
    try:
        model = joblib.load(model_path)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

# -------------------------
# Custom Styling
# -------------------------
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">
<style>
[data-testid="stNumberInput"] button {
    display: none !important;
}

/* Sidebar Specific Styling */
section[data-testid="stSidebar"] {
    background-color: #ffffff; /* Clean white background */
    border-right: 1px solid #dee2e6;
}
/* Force dark text for sidebar elements */
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3, 
section[data-testid="stSidebar"] p, 
section[data-testid="stSidebar"] li, 
section[data-testid="stSidebar"] span, 
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label {
    color: #2c3e50 !important;
}

[data-testid="stSidebar"] [data-testid="stImage"] img {
    background-color: white;
    padding: 5px;
    border-radius: 5px;
    border: 1px solid #dee2e6;
}
.main {
    background-color: #f8f9fa;
    color: #212529 !important; /* Ensure text is dark on light background */
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
    color: white;
}
h1 {
    color: #2c3e50 !important;
}
h2, h3 {
    color: #34495e !important;
}
/* Fix for general text visibility if theme defaults are wrong */
p, label, li, .stMarkdown {
    color: #212529;
}
.info-box {
    background-color:#010008;
    color: white !important; /* Fix invisible text in info box */
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
    st.image("C:\\Users\\gigan\\OneDrive\\Pictures\\group5.png", width=100) # Placeholder construction icon
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
    st.markdown("---")
    st.write("This tool uses Machine Learning to assess construction project risks based on environmental, structural, and financial parameters.")

# -------------------------
# Main Header
# -------------------------
st.markdown("<h1><i class='fas fa-hard-hat custom-icon'></i> Construction Project Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    Welcome to the Risk Assessment Tool. Please input the project details below to evaluate the potential risk level.
    This system analyzes historical data to provide safety insights.
</div>
""", unsafe_allow_html=True)

# -------------------------
# Input Form
# -------------------------
# Removed st.form to allow real-time updates for Labor Hours calculation
st.markdown("<h3><i class='fas fa-folder-open custom-icon'></i> Project Information</h3>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    project_type = st.selectbox("Project Type", ["Tunnel", "Dam", "Building", "Road"])
with col2:
    location = st.selectbox("Location", ["Lagos", "Kano"])
with col3:
    weather = st.selectbox("Weather Condition", ["Sunny", "Cloudy", "Harmattan"])
with col4:
    anomaly = st.selectbox("Anomaly Detected?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

st.markdown("---")
st.markdown("<h3><i class='fas fa-coins custom-icon'></i> Financials & Timeline</h3>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

with c1:
    planned_cost = st.number_input("Planned Cost (₦)", min_value=0.0, step=1000.0, format="%.2f")
    st.caption(f"Formatted: ₦{planned_cost:,.2f}")
with c2:
    actual_cost = st.number_input("Actual Cost (₦)", min_value=0.0, step=1000.0, format="%.2f")
    st.caption(f"Formatted: ₦{actual_cost:,.2f}")
with c3:
    planned_duration = st.number_input("Planned Duration (Days)", min_value=0.0, step=1.0)
with c4:
    actual_duration = st.number_input("Actual Duration (Days)", min_value=0.0, step=1.0)

st.markdown("---")
st.markdown("<h3><i class='fas fa-cogs custom-icon'></i> Structural & Environmental Metrics</h3>", unsafe_allow_html=True)
m1, m2, m3 = st.columns(3)

with m1:
    vibration = st.number_input("Vibration Level (m/s²)", min_value=0.0, format="%.4f")
    crack_width = st.number_input("Crack Width (mm)", min_value=0.0, format="%.4f")
    load_capacity = st.number_input("Load Bearing Capacity (MPa)", min_value=1.0) # Assume min 1 for sliders usually, but number input ok

with m2:
    temperature = st.slider("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
    humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    air_quality = st.slider("Air Quality Index", min_value=0, max_value=500, value=100)

with m3:
    energy = st.number_input("Energy Consumption (kWh)", min_value=0.0)
    material = st.number_input("Material Usage (kg)", min_value=0.0)

st.markdown("---")
st.markdown("<h3><i class='fas fa-users-cog custom-icon'></i> Resources & Safety</h3>", unsafe_allow_html=True)
r1, r2, r3 = st.columns(3)

# Auto-calculate Labor Hours
labor_calculated = actual_duration * 8.0

with r1:
    labor = st.number_input("Labor Hours (Auto-calculated)", value=labor_calculated, disabled=True, help="Calculated as Actual Duration × 8 hours")
with r2:
    equipment = st.slider("Equipment Utilization (%)", min_value=0.0, max_value=100.0, value=80.0)

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("Analyze Risk Level")

# -------------------------
# Prediction Logic
# -------------------------
if predict_btn:
    with st.spinner("Analyzing project parameters..."):
        # Create DataFrame
        input_data = {
            "Project_Type": project_type,
            "Location": location,
            "Weather_Condition": weather,
            "Anomaly_Detected": anomaly,
            "Planned_Cost (Naira)": planned_cost,
            "Actual_Cost (Naira)": actual_cost,
            "Planned_Duration (Days)": planned_duration,
            "Actual_Duration (Days)": actual_duration,
            "Vibration_Level (m/s²)": vibration,
            "Crack_Width": crack_width,
            "Load_Bearing_Capacity (MPa)": load_capacity,
            "Temperature (celcius)": temperature,
            "Humidity (%)": humidity,
            "Air_Quality_Index": air_quality,
            "Energy_Consumption": energy,
            "Material_Usage": material,
            "Labor_Hours": labor,
            "Equipment_Utilization": equipment,
        }
        
        input_df = pd.DataFrame([input_data])

        # One-hot encoding
        input_df = pd.get_dummies(input_df)

        # Align with model features
        model_features = model.feature_names_in_
        input_df = input_df.reindex(columns=model_features, fill_value=0)

        # Predict
        prediction = model.predict(input_df)[0]

        st.divider()
        st.markdown("<h3><i class='fas fa-chart-line custom-icon'></i> Analysis Result</h3>", unsafe_allow_html=True)
        
        # Display logic based on prediction
        if "High" in str(prediction) or "Collapse" in str(prediction):
            st.markdown(f"""
            <div style="background-color: #f8d7da; color: #721c24; padding: 20px; border-radius: 10px; border-left: 5px solid #bd2130;">
                <h4 style="margin: 0;"><i class="fas fa-exclamation-triangle"></i> Prediction: {prediction}</h4>
                <p style="margin: 10px 0 0 0;">This project shows signs of high risk. Immediate structural audit recommended.</p>
            </div>
            """, unsafe_allow_html=True)
        elif "Medium" in str(prediction):
            st.markdown(f"""
            <div style="background-color: #fff3cd; color: #856404; padding: 20px; border-radius: 10px; border-left: 5px solid #ffc107;">
                <h4 style="margin: 0;"><i class="fas fa-exclamation-circle"></i> Prediction: {prediction}</h4>
                <p style="margin: 10px 0 0 0;">Moderate risk detected. Monitor vibration and load capacity closely.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #d4edda; color: #155724; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745;">
                <h4 style="margin: 0;"><i class="fas fa-check-circle"></i> Prediction: {prediction}</h4>
                <p style="margin: 10px 0 0 0;">The project parameters indicate a stable or low-risk condition.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### <i class='fas fa-wave-square custom-icon'></i> Structural Analysis Graph", unsafe_allow_html=True)
        
        # Generate synthetic wave data for visualization
        # Using Vibration Level to determine amplitude and frequency characteristics
        base_amp = vibration if vibration > 0.1 else 0.5
        
        # Time steps from 0 to 1 with 200 points
        t = np.linspace(0, 1, 200)
        
        # Frequency influenced by vibration level (higher vibration = more erratic/faster)
        freq = 4 + (base_amp * 0.2)
        
        # Generate Sine Wave
        wave = base_amp * np.sin(2 * np.pi * freq * t)
        
        # Add noise to simulate real-world sensor data
        noise = np.random.normal(0, base_amp * 0.1, t.shape)
        
        # Final Signal
        signal = wave + noise
        
        # Create DataFrame (kept for compatibility) and plot with Matplotlib
        chart_df = pd.DataFrame({
            "Time (s)": t,
            "Vibration Amplitude (m/s²)": signal
        })

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(chart_df["Time (s)"], chart_df["Vibration Amplitude (m/s²)"], color="#007bff", linewidth=1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Vibration Amplitude (m/s²)")
        ax.set_title("Simulated Vibration Signal")
        ax.grid(True, linestyle='--', alpha=0.4)

        st.pyplot(fig)
        st.caption("Figure: Simulated vibration sensor reading based on current project parameters.")
