import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the saved model, encoder, and scaler
model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Function to process new input data
def process_input_data(input_data):
    # Split categorical and numerical features
    categorical_features = input_data[['workclass', 'education', 'marital-status', 'occupation',
                                       'relationship', 'race', 'sex', 'country']]
    numerical_features = input_data[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                                      'hours-per-week']]

    # One-Hot Encode categorical features
    encoded_categorical = encoder.transform(categorical_features).toarray()

    # Scale numerical features
    scaled_numerical = scaler.transform(numerical_features)

    # Combine processed features
    processed_features = pd.concat(
        [pd.DataFrame(encoded_categorical), pd.DataFrame(scaled_numerical)],
        axis=1
    )

    return processed_features

# Function to make predictions
def predict(input_data):
    # Process the input data
    processed_features = process_input_data(input_data)

    # Make predictions
    predictions = model.predict(processed_features)

    return predictions

# Example usage
if __name__ == "__main__":
    # Example input data
    example_data = pd.DataFrame({
        'workclass': [' Private'],
        'education': [' Bachelors'],
        'marital-status': [' Never-married'],
        'occupation': [' Exec-managerial'],
        'relationship': [' Not-in-family'],
        'race': [' White'],
        'sex': [' Male'],
        'country': [' United-States'],
        'age': [35],
        'fnlwgt': [77516],
        'education-num': [13],
        'capital-gain': [15000],
        'capital-loss': [1000],
        'hours-per-week': [40]
    })

    # Get predictions
    predictions = predict(example_data)

    print("Predictions:", predictions)