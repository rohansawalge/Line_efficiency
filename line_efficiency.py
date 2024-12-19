import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# Provided data (same as before)
data = {
    "Station Pitch (m)": [12.5] * 100,  # Constant value for station pitch
    "Average Conveyor Speed": [1.7, 2.3, 1.9, 2.1, 0.0, 1.8, 2.0, 1.5, 1.6, 1.5, 1.7, 0.0, 1.6, 1.5, 1.5, 0.0, 1.8, 1.2, 0.0, 1.4, 1.5, 1.4, 1.6, 1.8, 1.7, 0.0, 1.4, 1.3, 1.5, 1.5, 1.2, 0.0, 0.0, 2.0, 1.9, 1.7, 1.7, 1.3, 0.0, 0.0, 2.2, 1.5, 1.3, 1.4, 1.6, 1.8, 0.0, 1.4, 1.5, 1.3, 1.4, 0.0, 1.1, 0.0, 1.4, 1.5, 1.3, 1.2, 1.2, 1.8, 1.1, 1.3, 0.0, 1.3, 1.6, 1.2, 1.1, 0.0, 1.4, 1.9, 1.5, 1.2, 1.7, 0.0, 0.0, 1.8, 1.6, 1.4, 1.4, 1.4, 1.5, 0.0, 1.6, 1.3, 1.2, 1.6, 1.5, 1.5, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.7, 1.9, 1.7, 2.3, 1.8, 0.0, 1.5, 1.5, 1.9, 1.3, 1.4, 1.8, 0.0, 1.6, 1.3, 0.0, 1.9, 1.8, 1.3, 0.0, 1.5, 1.8, 1.3],
    "Basis Cycle Time (BCT) (min)": [5.21, 5.92, 5.21, 4.91, 0.00, 4.86, 4.60, 4.46, 4.26, 4.48, 4.23, 0.00, 4.33, 4.29, 4.43, 0.00, 4.28, 4.10, 0.00, 4.08, 4.61, 4.68, 4.51, 4.41, 4.15, 0.00, 4.28, 4.58, 4.37, 4.21, 3.87, 0.00, 0.00, 4.85, 5.11, 4.95, 5.29, 4.24, 0.00, 0.00, 4.85, 4.30, 3.99, 3.96, 4.02, 4.12, 0.00, 4.26, 4.18, 4.53, 4.01, 0.00, 3.72, 0.00, 3.90, 3.88, 3.66, 3.60, 3.87, 4.09, 3.86, 3.82, 0.00, 4.09, 4.14, 4.07, 4.04, 0.00, 4.26, 4.46, 4.11, 4.06, 4.13, 0.00, 0.00, 4.03, 4.27, 4.50, 4.20, 4.85, 4.21, 0.00, 4.14, 4.00, 3.90, 3.81, 4.27, 5.16, 3.67, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 5.83, 6.12, 4.91, 5.82, 4.89, 0.00, 4.79, 4.44, 4.26, 4.42, 4.00, 4.18, 0.00, 4.63, 4.28, 0.00, 4.64, 4.93, 4.14, 0.00, 4.34, 4.46, 4.51],
    "Total Available Time (min)": [455] * 100,  # New feature
    "Actual Rollout Quantity (n)": [15, 16, 14, 17, 13, 15, 18, 14, 16, 15, 17, 19, 16, 15, 14, 14, 18, 13, 17, 16, 15, 14, 16, 18, 19, 14, 15, 13, 16, 17, 14, 15, 19, 18, 16, 15, 14, 13, 16, 17, 19, 15, 14, 16, 18, 19, 14, 15, 17, 13, 16, 16, 14, 15, 17, 18, 16, 15, 14, 19, 13, 16, 17, 15, 18, 14, 13, 17, 15, 19, 16, 14, 18, 15, 16, 19, 17, 14, 15, 13, 16, 18, 17, 15, 14, 19, 16, 13, 14, 15, 18, 16, 17, 14, 15, 19, 16, 13, 14, 15, 17, 16, 18, 14, 15, 19, 13, 16, 18, 17, 15, 14, 16, 19, 17, 14, 15, 16, 18, 13],  # New feature
    "Line Efficiency (%)": [17.17, 20.83, 16.02, 18.36, 0.00, 16.03, 18.22, 13.73, 14.98, 14.77, 15.80, 0.00, 15.23, 14.13, 13.64, 0.00, 16.91, 11.72, 0.00, 14.36, 15.19, 14.39, 15.85, 17.46, 17.32, 0.00, 14.09, 13.10, 15.38, 15.73, 11.92, 0.00, 0.00, 19.21, 17.98, 16.31, 16.27, 12.10, 0.00, 0.00, 20.24, 14.16, 12.26, 13.93, 15.89, 17.20, 0.00, 14.05, 15.60, 12.95, 14.09, 0.00, 11.44, 0.00, 14.55, 15.35, 12.89, 11.86, 11.92, 17.09, 11.02, 13.42, 0.00, 13.48, 16.38, 12.53, 11.53, 0.00, 14.04, 18.64, 14.46, 12.49, 16.32, 0.00, 0.00, 16.84, 15.96, 13.84, 13.83, 13.87, 14.79, 0.00, 15.46, 13.20, 12.01, 15.91, 15.01, 14.73, 11.30, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 16.65, 18.82, 16.18, 21.75, 17.19, 0.00, 14.75, 14.64, 17.79, 12.62, 14.08, 16.52, 0.00, 15.26, 13.17, 0.00, 19.37, 18.42, 12.75, 0.00, 15.25, 17.65, 12.89]
}

# Convert data into a DataFrame
df = pd.DataFrame(data)

# Preprocessing
def preprocess_data(df):
    for column in df.columns:
        df[column] = df[column].replace(0, df[column].mean())
    scaler = MinMaxScaler()
    df[['Average Conveyor Speed', 'Basis Cycle Time (BCT) (min)', 'Line Efficiency (%)']] = scaler.fit_transform(
        df[['Average Conveyor Speed', 'Basis Cycle Time (BCT) (min)', 'Line Efficiency (%)']]
    )
    return df

# Preprocess the dataset
df = preprocess_data(df)

# Features and target
X = df[['Station Pitch (m)', 'Average Conveyor Speed', 'Basis Cycle Time (BCT) (min)', 'Total Available Time (min)', 'Actual Rollout Quantity (n)']]
y = df['Line Efficiency (%)']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a regression model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Streamlit app
st.title("Line Efficiency Predictor")

# Switch for data viewing
view_data = st.selectbox("Select to view data:", ["None", "Training Data", "Testing Data"])

if view_data == "Training Data":
    st.subheader("Training Data")
    st.write(X_train.assign(Efficiency=y_train))

elif view_data == "Testing Data":
    st.subheader("Testing Data")
    st.write(X_test.assign(Efficiency=y_test))

# Switch for prediction
predict_switch = st.checkbox("Enable Prediction")

if predict_switch:
    st.subheader("Predict Line Efficiency")
    station_pitch = st.number_input("Station Pitch (m)", value=12.5)
    avg_conveyor_speed = st.number_input("Average Conveyor Speed", value=1.5)
    bct = st.number_input("Basis Cycle Time (BCT) (min)", value=4.5)
    total_available_time = st.number_input("Total Available Time (min)", value=455)
    actual_rollout_quantity = st.number_input("Actual Rollout Quantity (n)", value=15)

    # Make predictions
    if st.button("Predict"):
        input_data = pd.DataFrame({
            "Station Pitch (m)": [station_pitch],
            "Average Conveyor Speed": [avg_conveyor_speed],
            "Basis Cycle Time (BCT) (min)": [bct],
            "Total Available Time (min)": [total_available_time],
            "Actual Rollout Quantity (n)": [actual_rollout_quantity]
        })
        prediction = model.predict(input_data)
        st.success(f"Predicted Line Efficiency: {prediction[0]:.2f}")

# Option to retrain the model with new data
retrain_switch = st.checkbox("Enable Retraining")

if retrain_switch:
    st.subheader("Retrain Model with New Data")
    new_data = st.file_uploader("Upload CSV file with new data", type=["csv"])

    if new_data:
        new_df = pd.read_csv(new_data)
        new_df = preprocess_data(new_df)

        X_new = new_df[['Station Pitch (m)', 'Average Conveyor Speed', 'Basis Cycle Time (BCT) (min)', 'Total Available Time (min)', 'Actual Rollout Quantity (n)']]
        y_new = new_df['Line Efficiency (%)']

        # Retrain model
        model.fit(X_new, y_new)
        st.success("Model retrained successfully!")

# Option to download the updated model
download_switch = st.checkbox("Download Updated Model")

if download_switch:
    with open("updated_model.pkl", "wb") as file:
        pickle.dump(model, file)
    st.success("Updated model saved as 'updated_model.pkl'")

    # Provide download link
    st.download_button(
        label="Download Updated Model",
        data=open("updated_model.pkl", "rb"),
        file_name="updated_model.pkl",
        mime="application/octet-stream"
    )










