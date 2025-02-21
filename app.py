from flask import Flask, render_template, request
import numpy as np
import pickle

# Load trained model and scaler
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Define the order of features based on dataset columns
        feature_order = ["age", "gender", "glucose", "blood_pressure", "bmi", "insulin"]
        
        # Get user input in correct order
        features = [float(request.form[key]) for key in feature_order]

        # Scale input
        features = scaler.transform([features])

        # Make prediction
        prediction = model.predict(features)

        # Determine result
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction="Invalid Input! Please enter valid numbers.")

if __name__ == '__main__':
    app.run(debug=True)
