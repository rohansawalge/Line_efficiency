import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import pickle

# Provided data
data = {
    "Station Pitch (m)": [12.5] * 100,  # Constant value for station pitch
    "Average Conveyor Speed": [1.7, 2.3, 1.9, 2.1, 0.0, 1.8, 2.0, 1.5, 1.6, 1.5, 1.7, 0.0, 1.6, 1.5, 1.5, 0.0, 1.8, 1.2, 0.0, 1.4, 1.5, 1.4, 1.6, 1.8, 1.7, 0.0, 1.4, 1.3, 1.5, 1.5, 1.2, 0.0, 0.0, 2.0, 1.9, 1.7, 1.7, 1.3, 0.0, 0.0, 2.2, 1.5, 1.3, 1.4, 1.6, 1.8, 0.0, 1.4, 1.5, 1.3, 1.4, 0.0, 1.1, 0.0, 1.4, 1.5, 1.3, 1.2, 1.2, 1.8, 1.1, 1.3, 0.0, 1.3, 1.6, 1.2, 1.1, 0.0, 1.4, 1.9, 1.5, 1.2, 1.7, 0.0, 0.0, 1.8, 1.6, 1.4, 1.4, 1.4, 1.5, 0.0, 1.6, 1.3, 1.2, 1.6, 1.5, 1.5, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.7, 1.9, 1.7, 2.3, 1.8, 0.0, 1.5, 1.5, 1.9, 1.3, 1.4, 1.8, 0.0, 1.6, 1.3, 0.0, 1.9, 1.8, 1.3, 0.0, 1.5, 1.8, 1.3],
    "Basis Cycle Time (BCT) (min)": [5.21, 5.92, 5.21, 4.91, 0.00, 4.86, 4.60, 4.46, 4.26, 4.48, 4.23, 0.00, 4.33, 4.29, 4.43, 0.00, 4.28, 4.10, 0.00, 4.08, 4.61, 4.68, 4.51, 4.41, 4.15, 0.00, 4.28, 4.58, 4.37, 4.21, 3.87, 0.00, 0.00, 4.85, 5.11, 4.95, 5.29, 4.24, 0.00, 0.00, 4.85, 4.30, 3.99, 3.96, 4.02, 4.12, 0.00, 4.26, 4.18, 4.53, 4.01, 0.00, 3.72, 0.00, 3.90, 3.88, 3.66, 3.60, 3.87, 4.09, 3.86, 3.82, 0.00, 4.09, 4.14, 4.07, 4.04, 0.00, 4.26, 4.46, 4.11, 4.06, 4.13, 0.00, 0.00, 4.03, 4.27, 4.50, 4.20, 4.85, 4.21, 0.00, 4.14, 4.00, 3.90, 3.81, 4.27, 5.16, 3.67, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 5.83, 6.12, 4.91, 5.82, 4.89, 0.00, 4.79, 4.44, 4.26, 4.42, 4.00, 4.18, 0.00, 4.63, 4.28, 0.00, 4.64, 4.93, 4.14, 0.00, 4.34, 4.46, 4.51],
    "Total Available Time (min)": [455] * 100,  # New feature
    "Actual Rollout Quantity (n)": [15, 16, 14, 17, 13, 15, 18, 14, 16, 15, 17, 19, 16, 15, 14, 14, 18, 13, 17, 16, 15, 14, 16, 18, 19, 14, 15, 13, 16, 17, 14, 15, 19, 18, 16, 15, 14, 13, 16, 17, 19, 15, 14, 16, 18, 19, 14, 15, 17, 13, 16, 16, 14, 15, 17, 18, 16, 15, 14, 19, 13, 16, 17, 15, 18, 14, 13, 17, 15, 19, 16, 14, 18, 15, 16, 19, 17, 14, 15, 13, 16, 18, 17, 15, 14, 19, 16, 13, 14, 15, 18, 16, 17, 14, 15, 19, 16, 13, 14, 15, 17, 16, 18, 14, 15, 19, 13, 16, 18, 17, 15, 14, 16, 19, 17, 14, 15, 16, 18, 13],  # New feature
    "Line Efficiency (%)": [17.17, 20.83, 16.02, 18.36, 0.00, 16.03, 18.22, 13.73, 14.98, 14.77, 15.80, 0.00, 15.23, 14.13, 13.64, 0.00, 16.91, 11.72, 0.00, 14.36, 15.19, 14.39, 15.85, 17.46, 17.32, 0.00, 14.09, 13.10, 15.38, 15.73, 11.92, 0.00, 0.00, 19.21, 17.98, 16.31, 16.27, 12.10, 0.00, 0.00, 20.24, 14.16, 12.26, 13.93, 15.89, 17.20, 0.00, 14.05, 15.60, 12.95, 14.09, 0.00, 11.44, 0.00, 14.55, 15.35, 12.89, 11.86, 11.92, 17.09, 11.02, 13.42, 0.00, 13.48, 16.38, 12.53, 11.53, 0.00, 17.46, 17.07, 16.56, 13.82, 0.00, 15.74, 19.06, 17.25, 16.88, 14.02, 0.00, 0.00, 16.29, 16.76, 16.73, 17.01, 15.93, 15.89, 0.00, 16.26, 17.24, 16.46, 17.68, 16.06, 16.28, 16.23, 0.00, 16.01, 16.56, 14.11, 17.17, 16.26],  # Target variable
}

# Ensure all lists have the same length (100)
required_length = 100
for key, value in data.items():
    if len(value) < required_length:
        data[key] = value + [None] * (required_length - len(value))
    elif len(value) > required_length:
        data[key] = value[:required_length]

# Create DataFrame
df = pd.DataFrame(data)

# Separate features and target
X = df.drop(columns=["Line Efficiency (%)"])
y = df["Line Efficiency (%)"]

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save the trained model
model_filename = "line_efficiency_model.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(model, f)

# Streamlit interface
st.title("AI-based Line Efficiency Prediction")

# Input fields for new data
station_pitch = st.number_input("Station Pitch (m)", value=12.5)
avg_conveyor_speed = st.number_input("Average Conveyor Speed", value=1.5)
bct = st.number_input("Basis Cycle Time (min)", value=4.5)
total_available_time = st.number_input("Total Available Time (min)", value=455)
actual_rollout_quantity = st.number_input("Actual Rollout Quantity (n)", value=15)

# Predict button
if st.button("Predict"):
    new_data = pd.DataFrame({
        "Station Pitch (m)": [station_pitch],
        "Average Conveyor Speed": [avg_conveyor_speed],
        "Basis Cycle Time (min)": [bct],
        "Total Available Time (min)": [total_available_time],
        "Actual Rollout Quantity (n)": [actual_rollout_quantity]
    })

    # Ensure the new data columns match the training data columns
    new_data = new_data[X.columns]

    # Scale new input data
    new_data_scaled = scaler.transform(new_data)

    # Load the saved model
    with open(model_filename, "rb") as f:
        loaded_model = pickle.load(f)

    # Make the prediction
    prediction = loaded_model.predict(new_data_scaled)
    st.write(f"Predicted Line Efficiency: {prediction[0]:.2f}%")

# Retrain model on new data
if st.button("Retrain Model with New Data"):
    # Retrain with the same training data
    model.fit(X_train, y_train)
    # Save the updated model
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    st.write("Model has been retrained and updated.")

# Download updated model
with open(model_filename, "rb") as f:
    st.download_button(label="Download Updated Model", data=f, file_name="updated_line_efficiency_model.pkl")













