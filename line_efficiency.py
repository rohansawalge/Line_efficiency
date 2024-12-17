import streamlit as st
import pandas as pd
import joblib  # For loading the AI model

# Load the trained AI model
model = joblib.load("updated_rf_model.pkl")  # Replace with your actual model file name

# App title and description
st.title("Line Efficiency Prediction Tool")
st.write("""
This app predicts **line efficiency** using a trained AI model. 
Enter the required inputs below to get started.
""")

# Input features
st.sidebar.header("Enter Input Features")
station_pitch = st.sidebar.number_input("Station Pitch", value=10.0)
total_attendance_planned = st.sidebar.number_input("Total Attendance Planned", value=100)
total_attendance_present = st.sidebar.number_input("Total Attendance Present", value=95)
difference = st.sidebar.number_input("Difference (Planned - Present)", value=5)
conveyor_speed = st.sidebar.number_input("Conveyor Speed", value=20.0)
BCT = st.sidebar.number_input("BCT (Balanced Cycle Time)", value=1.5)
DRR = st.sidebar.number_input("DRR (Daily Run Rate)", value=300)
total_available_time = st.sidebar.number_input("Total Available Time", value=480.0)
line_efficiency = st.sidebar.number_input("Line Efficiency (%)", value=85.0)
loss_time = st.sidebar.number_input("Loss Time", value=30.0)
line_availability = st.sidebar.number_input("Line Availability (%)", value=95.0)
SMH = st.sidebar.number_input("SMH (Standard Man Hours)", value=60.0)
working_days = st.sidebar.number_input("Working Days", value=22)
line_formation_ratio = st.sidebar.number_input("Line Formation Ratio", value=0.9)
min_efficiency = st.sidebar.number_input("Minimum Efficiency (%)", value=80.0)

# Create a DataFrame for the inputs
input_data = pd.DataFrame({
    'Station_pitch': [station_pitch],
    'Total_attendance_planned': [total_attendance_planned],
    'total_attendance_present': [total_attendance_present],
    'Difference': [difference],
    'conveyor_speed': [conveyor_speed],
    'BCT': [BCT],
    'DRR': [DRR],
    'total_available_time': [total_available_time],
    'line_efficiency': [line_efficiency],
    'loss_time': [loss_time],
    'line_availability': [line_availability],
    'SMH': [SMH],
    'working_days': [working_days],
    'line_formation_ratio': [line_formation_ratio],
    'min_efficiency': [min_efficiency]
})

# Display the inputs for user confirmation
st.write("### Input Data:")
st.write(input_data)

# Make predictions
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)  # Ensure your model is compatible with this input format
        st.success(f"Predicted Line Efficiency: {prediction[0]:.2f}%")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")




