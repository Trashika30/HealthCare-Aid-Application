from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the model and LabelEncoder
with open('diseasepredict.pkl', 'rb') as file:
    model = pickle.load(file)
with open('labelencoder.pkl', 'rb') as file:
    le = pickle.load(file)

@app.route('/', methods=['GET'])
def index():
    return render_template('dise.html')

@app.route('/submit-form', methods=['POST'])
def submit_form():
    
        # Get the input values from the form
        symptoms = [
            int(request.form['diarrhoea']),
            int(request.form['irritability']),
            int(request.form['dark_urine']),
            int(request.form['cough']),
            int(request.form['sweating']),
            int(request.form['itching']),
            int(request.form['joint_pain']),
            int(request.form['chest_pain']),
            int(request.form['malaise']),
            int(request.form['chills']),
            int(request.form['skin_rash']),
            int(request.form['yellowing_of_eyes']),
            int(request.form['yellowish_skin']),
            int(request.form['loss_of_appetite']),
            int(request.form['abdominal_pain']),
            int(request.form['headache']),
            int(request.form['nausea']),
            int(request.form['high_fever']),
            int(request.form['vomiting']),
            int(request.form['fatigue'])
        ]

        # Convert symptoms to numpy array for model prediction
        symptoms_array = np.array([symptoms])

        # Make a prediction with the loaded model
        predicted_index = model.predict(symptoms_array)[0]

        # Get the predicted class label (disease name)
        predicted_label = le.inverse_transform([predicted_index])[0]

        # Render the result.html template with the predicted output
        return render_template('dise.html', prediction=predicted_label)


if __name__ == "__main__":
    app.run(debug=True)
