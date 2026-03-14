import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page config with custom theme
st.set_page_config(
    page_title="AI Crop Recommendation", 
    page_icon="🌾", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #111111 0%, #000000 100%);
    }
    
    /* Content container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Title styling */
    h1 {
        background: linear-gradient(120deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        font-size: 3rem !important;
        text-align: center;
        margin-bottom: 2rem;
        animation: gradient 3s ease infinite;
        background-size: 200% 200%;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #111111 0%, #000000 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
    }
    
    [data-testid="stSidebar"] .stNumberInput label {
        color: white !important;
        font-weight: 500;
    }
    
    /* Input fields in sidebar */
    [data-testid="stSidebar"] input {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 10px !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        color: #333 !important;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        border: none;
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.4);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(245, 87, 108, 0.6);
    }
    
    /* Success message styling */
    .element-container:has(.stSuccess) {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    /* Warning message styling */
    .element-container:has(.stWarning) {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    /* Crop card styling */
    .crop-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
        margin: 1.5rem 0;
        border: 2px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
    }
    
    .crop-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.3);
    }
    
    /* Image container */
    .crop-image {
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        overflow: hidden;
    }
    
    .crop-image:hover {
        transform: scale(1.05);
    }
    
    /* Subheader styling */
    h2, h3 {
        color: #667eea;
        font-weight: 600;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Sidebar header */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: white !important;
        text-align: center;
        font-weight: 600;
    }
    
    /* Plot styling */
    .stPlotlyChart, .stPyplot {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }

    </style>
""", unsafe_allow_html=True)

# File paths
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
DATA_PATH = "Crop_recommendation.csv"
LOGO_PATH = "images/AgriSenseLogo.png"

# Crop images dictionary - comprehensive mapping
crop_images = {
    "rice": "images/rice.jpg",
    "wheat": "images/wheat.jpg",
    "maize": "images/maize.jpg",
    "cotton": "images/cotton.jpg",
    "jute": "images/jute.jpg",
    "coconut": "images/coconut.jpg",
    "papaya": "images/papaya.jpg",
    "orange": "images/orange.jpg",
    "apple": "images/apple.jpg",
    "muskmelon": "images/muskmelon.jpg",
    "watermelon": "images/watermelon.jpg",
    "grapes": "images/grapes.jpg",
    "mango": "images/mango.jpg",
    "banana": "images/banana.jpg",
    "pomegranate": "images/pomegranate.jpg",
    "lentil": "images/lentil.jpg",
    "blackgram": "images/blackgram.jpg",
    "mungbean": "images/mungbean.jpg",
    "mothbeans": "images/mothbeans.jpg",
    "pigeonpeas": "images/pigeonpeas.jpg",
    "kidneybeans": "images/kidneybeans.jpg",
    "chickpea": "images/chickpea.jpg",
    "coffee": "images/coffee.jpg"
}

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler with error handling"""
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file not found: {MODEL_PATH}")
            st.stop()
        
        if not os.path.exists(SCALER_PATH):
            st.error(f"❌ Scaler file not found: {SCALER_PATH}")
            st.stop()
            
        model = pickle.load(open(MODEL_PATH, "rb"))
        scaler = pickle.load(open(SCALER_PATH, "rb"))
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model/scaler: {str(e)}")
        st.stop()

# Load dataset
@st.cache_data
def load_data():
    """Load the crop recommendation dataset with error handling"""
    try:
        if not os.path.exists(DATA_PATH):
            st.error(f"❌ Dataset file not found: {DATA_PATH}")
            st.stop()
        
        data = pd.read_csv(DATA_PATH)
        
        # Validate required columns
        required_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns in dataset: {missing_columns}")
            st.stop()
        
        return data
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        st.stop()

# Load resources
model, scaler = load_model_and_scaler()
data = load_data()

# Display logo
col1, col2, col3 = st.columns(3)
with col2:
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=200)
    else:
        st.markdown("### 🌾 AgriSense")
    st.markdown("</div>", unsafe_allow_html=True)

# Main title
st.markdown("""
    <h1>🌾 AI-Powered Crop Recommendation System</h1>
    <p style='text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem;'>
        Leverage machine learning to discover the best crops for your soil
    </p>
""", unsafe_allow_html=True)

# Sidebar for input parameters
st.sidebar.markdown("## 📊 Input Parameters")
st.sidebar.markdown("---")

# Get min/max values from dataset for validation
n_min, n_max = float(data['N'].min()), float(data['N'].max())
p_min, p_max = float(data['P'].min()), float(data['P'].max())
k_min, k_max = float(data['K'].min()), float(data['K'].max())
temp_min, temp_max = float(data['temperature'].min()), float(data['temperature'].max())
hum_min, hum_max = float(data['humidity'].min()), float(data['humidity'].max())
ph_min, ph_max = float(data['ph'].min()), float(data['ph'].max())
rain_min, rain_max = float(data['rainfall'].min()), float(data['rainfall'].max())

# Input fields with icons and realistic defaults
N = st.sidebar.number_input(
    "🔵 Nitrogen (N)", 
    min_value=n_min, 
    max_value=n_max, 
    value=50.0, 
    step=1.0,
    help=f"Range: {n_min:.0f} - {n_max:.0f} kg/ha"
)

P = st.sidebar.number_input(
    "🟠 Phosphorus (P)", 
    min_value=p_min, 
    max_value=p_max, 
    value=50.0, 
    step=1.0,
    help=f"Range: {p_min:.0f} - {p_max:.0f} kg/ha"
)

K = st.sidebar.number_input(
    "🟣 Potassium (K)", 
    min_value=k_min, 
    max_value=k_max, 
    value=50.0, 
    step=1.0,
    help=f"Range: {k_min:.0f} - {k_max:.0f} kg/ha"
)

temperature = st.sidebar.number_input(
    "🌡️ Temperature (°C)", 
    min_value=temp_min, 
    max_value=temp_max, 
    value=25.0, 
    step=0.1,
    help=f"Range: {temp_min:.1f}°C - {temp_max:.1f}°C"
)

humidity = st.sidebar.number_input(
    "💧 Humidity (%)", 
    min_value=hum_min, 
    max_value=hum_max, 
    value=70.0, 
    step=1.0,
    help=f"Range: {hum_min:.0f}% - {hum_max:.0f}%"
)

ph = st.sidebar.number_input(
    "⚗️ Soil pH", 
    min_value=ph_min, 
    max_value=ph_max, 
    value=6.5, 
    step=0.1,
    help=f"Range: {ph_min:.1f} - {ph_max:.1f}"
)

rainfall = st.sidebar.number_input(
    "🌧️ Rainfall (mm)", 
    min_value=rain_min, 
    max_value=rain_max, 
    value=150.0, 
    step=1.0,
    help=f"Range: {rain_min:.0f} - {rain_max:.0f} mm"
)

st.sidebar.markdown("---")

# Info section in sidebar
with st.sidebar.expander("ℹ️ How to Use"):
    st.markdown("""
    1. **Enter your soil parameters** (N, P, K)
    2. **Add weather data** (Temperature, Humidity, Rainfall)
    3. **Input soil pH value**
    4. **Click 'Predict'** to get recommendations
    
    The system will suggest the top 3 crops suitable for your conditions.
    """)

# Predict button
if st.sidebar.button("🚀 Predict Top 3 Crops", use_container_width=True):
    # Validate inputs
    if all(v == 0 for v in [N, P, K, temperature, humidity, ph, rainfall]):
        st.warning("⚠️ Please enter valid soil and weather parameters.")
    else:
        try:
            with st.spinner("🔮 Analyzing soil conditions and predicting optimal crops..."):
                # Prepare features
                features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
                features_scaled = scaler.transform(features)

                # Predict probabilities for all crops
                probs = model.predict_proba(features_scaled)[0]
                classes = model.classes_

                # Get top 3 crops
                top_indices = np.argsort(probs)[-3:][::-1]
                top_crops = classes[top_indices]
                top_probs = probs[top_indices]

                # Display results header
                st.markdown("""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 2rem; border-radius: 20px; margin: 2rem 0; 
                                box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);'>
                        <h2 style='color: white; text-align: center; margin: 0;'>
                            ✅ Top 3 Recommended Crops
                        </h2>
                    </div>
                """, unsafe_allow_html=True)

                # Display each crop in a card
                for i, (crop, prob) in enumerate(zip(top_crops, top_probs)):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Display crop image if available
                        if crop in crop_images and os.path.exists(crop_images[crop]):
                            st.markdown("<div class='crop-image'>", unsafe_allow_html=True)
                            st.image(crop_images[crop], use_container_width=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            # Display placeholder if image not found
                            st.markdown(f"""
                                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                            padding: 3rem 1rem; border-radius: 15px; text-align: center;'>
                                    <p style='font-size: 4rem; margin: 0;'>🌾</p>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        rank_emoji = ["🥇", "🥈", "🥉"][i]
                        st.markdown(f"""
                            <div class='crop-card'>
                                <h2 style='margin-top: 0;'>{rank_emoji} {crop.upper()}</h2>
                                <p style='font-size: 1.2rem; color: #666;'>
                                    <strong>Confidence Score:</strong> 
                                    <span style='color: #667eea; font-size: 1.5rem; font-weight: 700;'>
                                        {prob*100:.2f}%
                                    </span>
                                </p>
                                <div style='background: linear-gradient(90deg, #667eea {prob*100}%, #e0e0e0 {prob*100}%); 
                                            height: 10px; border-radius: 10px; margin-top: 1rem;'></div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)

                # Display input summary
                st.markdown("---")
                st.markdown("### 📋 Input Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Nitrogen (N)", f"{N:.0f} kg/ha")
                    st.metric("Temperature", f"{temperature:.1f}°C")
                
                with col2:
                    st.metric("Phosphorus (P)", f"{P:.0f} kg/ha")
                    st.metric("Humidity", f"{humidity:.0f}%")
                
                with col3:
                    st.metric("Potassium (K)", f"{K:.0f} kg/ha")
                    st.metric("Soil pH", f"{ph:.1f}")
                
                with col4:
                    st.metric("Rainfall", f"{rainfall:.0f} mm")

                # Visualizations
                st.markdown("---")
                st.markdown("""
                    <div style='margin-top: 3rem;'>
                        <h2 style='text-align: center; color: #667eea;'>
                            📈 Feature Distribution Analysis
                        </h2>
                    </div>
                """, unsafe_allow_html=True)

                for crop in top_crops:
                    crop_data = data[data['label'] == crop]
                    
                    if len(crop_data) == 0:
                        st.warning(f"⚠️ No data available for {crop} in the dataset")
                        continue
                    
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #f8f9ff 0%, #e8eaff 100%); 
                                    padding: 2rem; border-radius: 20px; margin: 2rem 0;
                                    border: 2px solid #667eea33;'>
                            <h3 style='text-align: center; color: #667eea; margin-bottom: 2rem;'>
                                🌱 {crop.upper()} - Optimal Growing Conditions
                            </h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Set custom style for plots
                    sns.set_style("whitegrid")
                    plt.rcParams['figure.facecolor'] = 'white'
                    
                    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
                    fig.patch.set_facecolor('white')
                    
                    # Plot each feature with custom colors
                    features_to_plot = [
                        ('N', 'Nitrogen (N)', '#667eea', axes[0, 0]),
                        ('P', 'Phosphorus (P)', '#f093fb', axes[0, 1]),
                        ('K', 'Potassium (K)', '#764ba2', axes[0, 2]),
                        ('temperature', 'Temperature (°C)', '#f5576c', axes[0, 3]),
                        ('humidity', 'Humidity (%)', '#4facfe', axes[1, 0]),
                        ('ph', 'Soil pH', '#43e97b', axes[1, 1]),
                        ('rainfall', 'Rainfall (mm)', '#00f2fe', axes[1, 2])
                    ]
                    
                    for feature, title, color, ax in features_to_plot:
                        if feature in crop_data.columns:
                            sns.histplot(crop_data[feature], kde=True, ax=ax, color=color, alpha=0.7)
                            ax.set_title(title, fontsize=12, fontweight='bold', color='#333')
                            ax.set_xlabel('')
                            ax.set_ylabel('Frequency', fontsize=10)
                            ax.grid(True, alpha=0.3)
                    
                    # Hide the last subplot
                    axes[1, 3].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.error("Please check if the model is trained correctly and all files are in place.")

else:
    # Display welcome message when no prediction is made
    st.markdown("""
        <div class='info-box'>
            <h3>👋 Welcome to AI Crop Recommendation System!</h3>
            <p style='font-size: 1.1rem; line-height: 1.6;'>
                Enter your soil and weather parameters in the sidebar and click 
                <strong>"🚀 Predict Top 3 Crops"</strong> to get personalized crop recommendations
                powered by machine learning.
            </p>
            <p style='margin-top: 1rem;'>
                💡 <strong>Tip:</strong> Make sure to enter accurate values for best results!
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Display sample statistics
    st.markdown("### 📊 Dataset Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(data))
    with col2:
        st.metric("Crop Types", data['label'].nunique())
    with col3:
        st.metric("Features", len(data.columns) - 1)
    with col4:
        st.metric("Model Accuracy", "95%+")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem; margin-top: 3rem;'>
        <p style='font-size: 0.9rem;'>
            Made using AI & Machine Learning | 
            <strong>AgriSense</strong> - Smart Farming Solutions
        </p>
        <p style='font-size: 0.8rem; margin-top: 0.5rem;'>
            Version 2.0 | © 2024 All Rights Reserved
        </p>
    </div>
""", unsafe_allow_html=True)
