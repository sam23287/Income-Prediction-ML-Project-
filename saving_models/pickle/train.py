from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import pickle
from scipy.stats import boxcox, skew

# Load the dataset
data = pd.read_csv('D:\ML Project (Income Prediction)\income.csv')
print(data.head())

# Function to apply transformations based on skewness
def transform_skewed_data(df):
    for col in ['capital-gain', 'capital-loss']:
        col_skewness = skew(df[col].dropna())
        
        if col_skewness > 1:  # Right-skewed (Positive skewness)
            df[col] = np.log(df[col] + 1e-5)  # Handle zero or negative values
            print(f"Applied log transformation to right-skewed column: {col}")
        
        elif col_skewness < -1:  # Left-skewed (Negative skewness)
            df[col] = np.log(df[col].max() + 1 - df[col])
            print(f"Applied reflect and log transformation to left-skewed column: {col}")
        
        else:  # Both-sided skewed or near-symmetric
            if (df[col] <= 0).any():
                df[col] = df[col] - df[col].min() + 1  # Shift values to be positive
            df[col], _ = boxcox(df[col])
            print(f"Applied Box-Cox transformation to column with both-sided skew or near-symmetry: {col}")

    return df

# Apply transformations
df = transform_skewed_data(data)

# Split features and target
categorical_features = df[['workclass', 'education', 'marital-status', 'occupation',
                           'relationship', 'race', 'sex', 'country']]
numerical_features = df[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                         'hours-per-week']]
target = data['salary']

# One-Hot Encoding for categorical features
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_categorical = encoder.fit_transform(categorical_features).toarray()

# Scaling for numerical features
scaler = StandardScaler()
scaled_numerical = scaler.fit_transform(numerical_features)

# Combine processed features
processed_features = pd.concat(
    [pd.DataFrame(encoded_categorical), pd.DataFrame(scaled_numerical)],
    axis=1
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(processed_features, target, test_size=0.2, random_state=42)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"Resampled training data shape: {X_train_resampled.shape}")

# Train the model with resampled data
model = GradientBoostingClassifier()
model.fit(X_train_resampled, y_train_resampled)

# Evaluate the model
y_test_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred)
print(f'Testing accuracy: {accuracy}')

# Save the model, encoder, and scaler
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(encoder, open('encoder.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))