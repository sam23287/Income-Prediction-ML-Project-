from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

# Load the saved model, encoder, and scaler
model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__,template_folder=r'C:\Users\Shivam\Desktop\ML Project (Income Prediction)\deployment\templates')

# Define the home route
@app.route('/')
def home():
    return render_template('home.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    input_data = {
        'workclass': [request.form.get('workclass')],
        'education': [request.form.get('education')],
        'marital-status': [request.form.get('marital-status')],
        'occupation': [request.form.get('occupation')],
        'relationship': [request.form.get('relationship')],
        'race': [request.form.get('race')],
        'sex': [request.form.get('sex')],
        'country': [request.form.get('country')],
        'age': [int(request.form.get('age', 0))],
        'fnlwgt': [int(request.form.get('fnlwgt', 0))],
        'education-num': [int(request.form.get('education-num', 0))],
        'capital-gain': [int(request.form.get('capital-gain', 0))],
        'capital-loss': [int(request.form.get('capital-loss', 0))],
        'hours-per-week': [int(request.form.get('hours-per-week', 0))]
    }

    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)

    # Process input data
    categorical_features = input_df[['workclass', 'education', 'marital-status', 'occupation',
                                     'relationship', 'race', 'sex', 'country']]
    numerical_features = input_df[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                                    'hours-per-week']]

    # Encode and scale
    encoded_categorical = encoder.transform(categorical_features).toarray()
    scaled_numerical = scaler.transform(numerical_features)

    # Combine features
    processed_features = pd.concat(
        [pd.DataFrame(encoded_categorical), pd.DataFrame(scaled_numerical)],
        axis=1
    )

    # Make prediction
    prediction = model.predict(processed_features)[0]

    if prediction==0:
        a='less than 50K'
    else:
        a='greater than 50K'

    # Return the result
    return render_template('home.html', prediction_text=f'The predicted salary class is: {a}')

if __name__ == "__main__":
    app.run(debug=True)
