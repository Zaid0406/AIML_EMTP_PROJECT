from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("best_model.pkl")

# Home Page
@app.route("/")
def home():
    return render_template("home.html")

# Manual Input Page
@app.route("/manual")
def manual():
    return render_template("Manual_predict.html")

# Sensor Input Page
@app.route("/sensor")
def sensor():
    return render_template("Sensor_predict.html")

# Manual Prediction Route
@app.route("/predict_manual", methods=["POST"])
def predict_manual():
    feature1 = float(request.form["feature1"])
    feature2 = float(request.form["feature2"])
    feature3 = float(request.form["feature3"])

    features = np.array([[feature1, feature2, feature3]])
    prediction = model.predict(features)[0]

    return render_template("Manual_predict.html", prediction=round(prediction, 2))

# Sensor Prediction Route
@app.route("/predict_sensor", methods=["POST"])
def predict_sensor():
    sensor1 = float(request.form["sensor1"])
    sensor2 = float(request.form["sensor2"])
    sensor3 = float(request.form["sensor3"])

    features = np.array([[sensor1, sensor2, sensor3]])
    prediction = model.predict(features)[0]

    return render_template("Sensor_predict.html", prediction=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)
