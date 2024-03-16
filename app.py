from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('try.html')



@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    dissolved_oxygen = float(request.form['dissolved_oxygen'])
    ph = float(request.form['ph'])
    conductivity = float(request.form['conductivity'])
    bod = float(request.form['bod'])
    nitrate = float(request.form['nitrate'])
    total_coliform = float(request.form['total_coliform'])
    year = int(request.form['year'])
    
    # Prepare the input features as a NumPy array
    features = np.array([[dissolved_oxygen, ph, conductivity, bod, nitrate, total_coliform, year]])
    
    # Predict water quality using the model
    prediction = model.predict(features)
    
    # Determine the water quality assessment
    if prediction <= 25:
        assessment = "Very poor water quality"
    elif prediction <= 49:
        assessment = "Poor water quality"
    elif prediction <= 70:
        assessment = "Medium water quality"
    elif prediction <= 89:
        assessment = "Good water quality"
    else:
        assessment = "Excellent water quality"
    return render_template('pages/try.html', prediction=prediction, assessment=assessment)

if __name__ == '__main__':
    app.run(debug=True)

app = Flask(__name__)

