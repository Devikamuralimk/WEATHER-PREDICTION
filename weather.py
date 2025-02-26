import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64

# Function to convert image to Base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Load and encode background image
image_base64 = get_base64_image("anao-extreme-weather-BOM.jpg")

# Apply CSS for background image and text styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{image_base64}");
        background-size: cover;
    }}
    h1, h2, h3, h4, h5, h6, p, label {{
        color: white !important;
    }}
    .stSuccess {{
        background-color: #f0f0f0 !important;
        color: black !important;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
    }}
    /* Styling for the Predict button */
    div.stButton > button {{
        color: black !important;
        background-color: #ffcc00 !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        padding: 10px 15px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.title("🌤️ AI-Powered Weather Predictor")
st.write("This app predicts the weather type based on input features.")

# Load the saved model and encoder
with open('main.pkl', 'rb') as file:
    model_data = pickle.load(file)

model = model_data['model']
one_hot = model_data['one_hot']

# Get categories from encoder
season_categories = one_hot.categories_[0].tolist()
location_categories = one_hot.categories_[1].tolist()

# User input fields
st.subheader("Enter the weather parameters below:")
Temperature = st.number_input("🌡️ Temperature (°C)", value=20.0, step=0.1)
Humidity = st.number_input("💧 Humidity (%)", value=50.0, step=0.1)
Wind_Speed = st.number_input("🌬️ Wind Speed (km/h)", value=10.0, step=0.1)
Pressure = st.number_input("📊 Pressure (hPa)", value=1013.0, step=0.1)
Cloud_Cover = st.selectbox("☁️ Cloud Cover", ['clear', 'partly cloudy', 'cloudy', 'overcast'])
Season = st.selectbox("📅 Season", season_categories)
Location = st.selectbox("📍 Location", location_categories)

# Encode categorical inputs
cloud_cover_mapping = {'clear': 0, 'partly cloudy': 1, 'cloudy': 2, 'overcast': 3}
Cloud_Cover = cloud_cover_mapping[Cloud_Cover]

# Encode season and location using one-hot encoding
df_input = pd.DataFrame([[Season, Location]], columns=['Season', 'Location'])
df_encoded = pd.DataFrame(one_hot.transform(df_input), columns=one_hot.get_feature_names_out())

# Combine features
input_features = np.array([[Temperature, Humidity, Wind_Speed, Pressure, Cloud_Cover]])
input_data = np.hstack((input_features, df_encoded.values))

# Ensure correct feature dimensions
expected_features = model.n_features_in_
if input_data.shape[1] < expected_features:
    missing_features = expected_features - input_data.shape[1]
    input_data = np.hstack((input_data, np.zeros((1, missing_features))))
elif input_data.shape[1] > expected_features:
    input_data = input_data[:, :expected_features]

# Predict button
if st.button("🔍 Predict Weather Type"):
    try:
        prediction = model.predict(input_data)[0]
        labels = {0: '❄️ Snowy', 1: '🌧️ Rainy', 2: '☁️ Cloudy', 3: '☀️ Sunny'}
        
        # Custom-styled prediction box
        st.markdown(f'<div class="stSuccess">Predicted Weather Type: {labels[prediction]}</div>', unsafe_allow_html=True)
        
        # Display animations based on prediction
        if prediction == 0:
            st.snow()
        elif prediction == 1:
            st.warning("🌧️ It might rain today. Carry an umbrella!")
        elif prediction == 2:
            st.info("☁️ It's cloudy today. No rain expected.")
        elif prediction == 3:
            st.balloons()  # Celebrate sunny weather

    except ValueError as e:
        st.error(f"⚠️ An error occurred: {e}")







