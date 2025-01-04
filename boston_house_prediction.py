# Install seaborn if not already installed
# Run the following command in your terminal or command prompt:
# pip install seaborn

# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the Boston Housing dataset
@st.cache_data
def load_data():
    # Preloaded Boston Housing dataset (replace 'HousingData.csv' with actual path if needed)
    dataset_path = "HousingData.csv"  # Make sure this file is in the same directory as the script
    data = pd.read_csv(dataset_path)
    return data

# Function to train and evaluate models
def train_models(X_train, X_test, y_train, y_test):
    # Train Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    # Train Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    return (lr_model, mse_lr, r2_lr, rf_model, mse_rf, r2_rf, y_pred_lr, y_pred_rf)

# Main Streamlit App
st.title("Boston House Price Prediction App")
st.markdown("""
This is a simple web application that predicts Boston house prices based on various features like crime rate, number of rooms, and more. The app uses both Linear Regression and Random Forest models.
""")

# Step 1: Load the dataset
st.header("Step 1: Load and Explore the Dataset")
data = load_data()

# Display the dataset overview
if st.checkbox("Show Dataset Preview"):
    st.subheader("Dataset Preview")
    st.write(data.head())

if st.checkbox("Show Dataset Information"):
    st.subheader("Dataset Information")
    st.write(data.info())
    st.write("Missing Values:", data.isnull().sum())

# Step 2: Handle Missing Values
st.header("Step 2: Handle Missing Values")
data.fillna(data.mean(), inplace=True)
st.write("Missing values have been handled by filling them with the mean.")

# Step 3: Exploratory Data Analysis (EDA)
st.header("Step 3: Exploratory Data Analysis (EDA)")
if st.checkbox("Show Correlation Heatmap"):
    st.subheader("Correlation Heatmap")
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    st.pyplot(plt)

if st.checkbox("Show Target Variable Distribution"):
    st.subheader("Distribution of Target Variable (MEDV)")
    plt.figure(figsize=(8, 6))
    sns.histplot(data['MEDV'], kde=True, bins=30, color='blue')
    plt.title("Distribution of MEDV (House Prices)")
    plt.xlabel("Median Value of Homes ($1000s)")
    plt.ylabel("Frequency")
    st.pyplot(plt)

# Step 4: Prepare the Data
st.header("Step 4: Prepare the Data")
X = data.drop(columns=['MEDV'])
y = data['MEDV']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
st.write("Training and testing datasets created.")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
st.write("Features have been standardized.")

# Step 5: Train Models
st.header("Step 5: Train Models")
if st.button("Train Models"):
    lr_model, mse_lr, r2_lr, rf_model, mse_rf, r2_rf, y_pred_lr, y_pred_rf = train_models(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    # Step 6: Evaluate Models
    st.header("Step 6: Evaluate Models")
    st.subheader("Linear Regression Performance")
    st.write(f"Mean Squared Error: {mse_lr:.2f}")
    st.write(f"R² Score: {r2_lr:.2f}")

    st.subheader("Random Forest Regressor Performance")
    st.write(f"Mean Squared Error: {mse_rf:.2f}")
    st.write(f"R² Score: {r2_rf:.2f}")

    # Step 7: Visualize Results
    st.header("Step 7: Visualize Results")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_lr, label="Linear Regression", alpha=0.7, color="blue")
    plt.scatter(y_test, y_pred_rf, label="Random Forest", alpha=0.7, color="green")
    plt.plot(
        [min(y_test), max(y_test)], 
        [min(y_test), max(y_test)], 
        'k--', 
        label="Perfect Prediction"
    )
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.legend()
    plt.title("Actual vs Predicted House Prices")
    st.pyplot(plt)

    # Step 8: Save the Best Model
    st.header("Step 8: Save the Best Model")
    if mse_rf < mse_lr:  # Save the Random Forest model if it's better
        joblib.dump(rf_model, 'boston_house_price_model.pkl')
        st.write("Random Forest model saved as 'boston_house_price_model.pkl'.")
    else:
        joblib.dump(lr_model, 'boston_house_price_model.pkl')
        st.write("Linear Regression model saved as 'boston_house_price_model.pkl'.")
