import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open('Line efficiency_rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title of the app
st.title("Predict Target Using Your ML Model")

# Subtitle
st.subheader("Enter the features to predict the target")

# Feature input fields
# Example: Assume your model has 3 features: 'Feature1', 'Feature2', 'Feature3'
# Add input fields for each feature
Station_pitch = st.number_input("Enter Station_pitch (e.g., numerical value)", value=0.0)
Total_attendance_planned = st.number_input("Enter Total_attendance_planned (e.g., numerical value)", value=0.0)
total_attendance_present = st.number_input("Enter total_attendance_present (e.g., numerical value)", value=0.0)
Difference = st.number_input("Enter Difference (e.g., numerical value)", value=0.0)
conveyor_speed = st.number_input("Enter conveyor_speed (e.g., numerical value)", value=0.0)
BCT = st.number_input("Enter BCT (e.g., numerical value)", value=0.0)
DRR = st.number_input("Enter DRR (e.g., numerical value)", value=0.0)
total_available time = st.number_input("Enter total_available time (e.g., numerical value)", value=0.0)
line_efficiency = st.number_input("Enter line_efficiency (e.g., numerical value)", value=0.0)
loss_time = st.number_input("Enter loss_time (e.g., numerical value)", value=0.0)
line_availability = st.number_input("Enter line_availability (e.g., numerical value)", value=0.0)
SMH = st.number_input("Enter SMH (e.g., numerical value)", value=0.0)
working_days = st.number_input("Enter working_days (e.g., numerical value)", value=0.0)
line_formation_ratio = st.number_input("Enter line_formation_ratio (e.g., numerical value)", value=0.0)
min_efficiency = st.number_input("Enter min_efficiency (e.g., numerical value)", value=0.0)
# Organize the inputs into a DataFrame
input_data = pd.DataFrame({
    'Station_pitch': [Station_pitch],
    'Total_attendance_planned': [Total_attendance_planned],
    'total_attendance_present': [total_attendance_present],
    'Difference': [Difference],
    'conveyor_speed': [conveyor_speed],
    'BCT': [BCT],
    'DRR': [DRR],
    'total_available time': [total_available time],
    'line_efficiency': [line_efficiency],
    'loss_time': [loss_time],
    'line_availability': [line_availability],
    'SMH': [SMH],
    'working_days': [working_days],
    'line_formation_ratio': [line_formation_ratio],
    'min_efficiency': [min_efficiency]
})

# Add a button to make predictions
if st.button("Predict"):
    # Perform prediction
    prediction = model.predict(input_data)

    # Display the output
    st.subheader("Prediction Result")
    st.write(f"The predicted target value is: {prediction[0]}")

# Add a sidebar for additional information
st.sidebar.header("About the App")
st.sidebar.write("This app allows you to input feature values and predicts the target using the trained ML model.")

