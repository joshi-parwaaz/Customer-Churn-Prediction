import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Function to load data, preprocess, train models, and store results
def train_models(data_path):
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Error: File not found at '{data_path}'. Please check the path and try again.")
        return None  # Indicate error to prevent training

    # Preprocessing (assuming no additional preprocessing needed)

    # Define features and target
    X = df.drop('Exited', axis=1)
    y = df['Exited']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate models
    models = {}

    # Logistic Regression
    models['Logistic Regression'] = {}
    try:
        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(X_train, y_train)
        y_pred_logreg = logreg.predict(X_test)
        models['Logistic Regression']['accuracy'] = accuracy_score(y_test, y_pred_logreg)
        models['Logistic Regression']['report'] = classification_report(y_test, y_pred_logreg)
        cm_logreg = confusion_matrix(y_test, y_pred_logreg)
        models['Logistic Regression']['confusion_matrix'] = cm_logreg
    except Exception as e:
        st.error(f"Logistic Regression training failed: {e}")

    # Random Forest
    models['Random Forest'] = {}
    try:
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        models['Random Forest']['accuracy'] = accuracy_score(y_test, y_pred_rf)
        models['Random Forest']['report'] = classification_report(y_test, y_pred_rf)
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        models['Random Forest']['confusion_matrix'] = cm_rf
    except Exception as e:
        st.error(f"Random Forest training failed: {e}")

    # K-Nearest Neighbors
    models['K-Nearest Neighbors'] = {}
    try:
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        models['K-Nearest Neighbors']['accuracy'] = accuracy_score(y_test, y_pred_knn)
        models['K-Nearest Neighbors']['report'] = classification_report(y_test, y_pred_knn)
        cm_knn = confusion_matrix(y_test, y_pred_knn)
        models['K-Nearest Neighbors']['confusion_matrix'] = cm_knn
    except Exception as e:
        st.error(f"KNN model training failed: {e}")

    return models

# Define the file path (using relative path)
data_path = 'Dataset.csv' 

# Use st.cache_data for caching
@st.cache_data(persist=True, show_spinner=True)  # Persist across app restarts, show spinner
def cached_models():
    models = train_models(data_path)
    if models is not None:  # Check if training succeeded
        return models
    else:
        return {}  # Return empty dict if training failed

# Load trained models from the backend
models = cached_models()

# Title and introduction
st.title("Customer Churn Prediction")
st.write("This app helps you predict customer churn using various machine learning models.")

# Debug: Display available models
st.write("Available Models:", list(models.keys()))

# Select model dropdown
if models:  # Ensure models are loaded successfully
    selected_model = st.selectbox("Select Model", options=list(models.keys()))

    # Display selected model's details
    st.write(f"**Selected Model:** {selected_model}")
    st.write(f"**Accuracy:** {models[selected_model]['accuracy']:.4f}")
    st.write(f"**Classification Report:**")
    st.code(models[selected_model]['report'], language="python")

    # Display Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = models[selected_model]['confusion_matrix']
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed', 'Exited'], yticklabels=['Stayed', 'Exited'], ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f"{selected_model} Confusion Matrix")
    st.pyplot(fig)
else:
    st.error("No models available. Please check the dataset or training process.")
