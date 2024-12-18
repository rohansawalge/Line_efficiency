import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the pre-trained model and scaler
@st.cache_resource
def load_model():
    return joblib.load("updated_rf_model.pkl")

@st.cache_resource
def load_scaler():
    # Dummy scaler initialization (replace if you have a pre-saved scaler)
    return StandardScaler()

model = load_model()
scaler = load_scaler()

# Feature names
feature_names = [
    'Station_pitch', 'Total_attendance_planned', 'total_attendance_present',
    'Difference', 'conveyor_speed', 'BCT', 'DRR', 'total_available_time',
    'line_efficiency', 'loss_time', 'line_availability', 'SMH',
    'working_days', 'line_formation_ratio', 'min_efficiency'
]

# UI Layout
st.title("Line Efficiency Prediction and Model Training")

# Initialize placeholders for model input
inputs = {}

# Input Form
st.header("Enter Data for Prediction and Training")

for feature in feature_names:
    inputs[feature] = st.number_input(f"Enter {feature}:", value=0.0)

# Predict Button
if st.button("Predict"):
    # Prepare input for prediction
    input_data = np.array([inputs[feature] for feature in feature_names]).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    
    # Predict and display result
    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted Equivalent Line Efficiency: {prediction:.2f}")

# Update Model Button
if st.button("Update Model with Entered Data"):
    # Convert inputs to array for model retraining
    input_data = np.array([inputs[feature] for feature in feature_names]).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    
    # Add new data to the training set
    new_X = input_scaled
    new_y = np.array([prediction])  # Use the predicted value as the target for retraining
    
    # Retrain the model with the new data
    model.fit(new_X, new_y)
    
    # Save the updated model
    joblib.dump(model, "updated_rf_model.pkl")
    st.success("Model updated successfully!")
    
    # Evaluate the updated model (with the new data only)
    y_pred = model.predict(new_X)
    mae = mean_absolute_error(new_y, y_pred)
    mse = mean_squared_error(new_y, y_pred)
    r2 = r2_score(new_y, y_pred)
    
    # Display performance metrics
    st.write("### Updated Model Performance (based on new data):")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")
    
    # Provide download option for updated model
    with open("updated_rf_model.pkl", "rb") as f:
        st.download_button(
            label="Download Updated Model",
            data=f,
            file_name="updated_rf_model.pkl",
            mime="application/octet-stream"
        )







