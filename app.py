import streamlit as st
import pickle
import numpy as np

# Define the normalization function
def normalize_prediction(predicted_value, original_min, original_max, target_min, target_max):
    """Normalize a prediction to the target range."""
    normalized_value = ((predicted_value - original_min) / (original_max - original_min)) * (target_max - target_min) + target_min
    return np.clip(normalized_value, target_min, target_max)  # Ensure the value is within the target range

# Load your model
with open('HappinessModel.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Define the range of your model's expected output
original_min = 0  # Minimum expected value before normalization
original_max = 20  # Maximum expected value before normalization

# Define the target range
target_min = 1
target_max = 10

# Streamlit app
st.title("Happiness Prediction Model")

# Get user inputs
feature1 = st.slider("How satisfied are you with your job? (Scale 1-10)", 1, 10, 5)
feature2 = st.slider("How satisfied are you with your work-life balance? (Scale 1-10)", 1, 10, 5)
feature3 = st.slider("How often do you engage in social activities? (Hours per week)", 0, 168, 0)
feature4 = st.slider("How would you rate your overall health? (Scale 1-10)", 1, 10, 5)
feature5 = st.slider("How secure do you feel in your current job? (Scale 1-10)", 1, 10, 5)
feature6 = st.slider("How often do you experience stress? (Scale 1-10)", 1, 10, 5)
feature7 = st.slider("How supportive is your family? (Scale 1-10)", 1, 10, 5)
feature8 = st.slider("How much do you enjoy your hobbies? (Scale 1-10)", 1, 10, 5)

input_features = [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8]

# Make prediction
predicted_value = pipeline.predict([input_features])[0]

# Normalize the prediction
normalized_prediction = normalize_prediction(predicted_value, original_min, original_max, target_min, target_max)

# Display results
st.write(f"The predicted value [raw]: {predicted_value:.2f}")
st.write(f"The normalized prediction value is: {normalized_prediction:.2f}")

if normalized_prediction <= 3:
    st.write("The prediction indicates a lower level of happiness or satisfaction.")
elif normalized_prediction <= 7:
    st.write("The prediction indicates a moderate level of happiness or satisfaction.")
else:
    st.write("The prediction indicates a higher level of happiness or satisfaction.")
