from flask import Flask, request, jsonify, render_template
import pickle as pkl
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
with open('model.pkl', 'rb') as file:
    regressor = pkl.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form and validate
        age = float(request.form['age'])
        sex = int(request.form['sex'])  # Radio button (1 for Male, 0 for Female)
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])  # Radio button (1 for Smoker, 0 for Non-smoker)
        region = int(request.form['region'])  # Dropdown (0-3)

        # Ensure all inputs are within expected ranges
        if not (18 <= age <= 100):
            return jsonify({'error': 'Age must be between 18 and 100'})
        if sex not in [0, 1]:
            return jsonify({'error': 'Sex must be 0 (Female) or 1 (Male)'})
        if not (10 <= bmi <= 50):
            return jsonify({'error': 'BMI must be between 10 and 50'})
        if children < 0:
            return jsonify({'error': 'Number of children cannot be negative'})
        if smoker not in [0, 1]:
            return jsonify({'error': 'Smoker must be 0 (Non-smoker) or 1 (Smoker)'})
        if region not in [0, 1, 2, 3]:
            return jsonify({'error': 'Region must be between 0 and 3'})
        
        # Prepare the input data for prediction
        input_data = np.array([age, sex, bmi, children, smoker, region]).reshape(1, -1)
        
        # Make prediction
        prediction = regressor.predict(input_data)[0]

        # Return the result as a JSON object
        return jsonify({'insurance_cost': round(prediction, 2)})

    except ValueError as ve:
        return jsonify({'error': str(ve)})

if __name__ == '__main__':
    app.run(debug=True)

