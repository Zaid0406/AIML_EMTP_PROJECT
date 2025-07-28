# AIML_EMTP_PROJECT

This project focuses on predicting the **temperature of an electric motor** using Artificial Intelligence and Machine Learning techniques.

## 🔧 Features
- Web interface built with **HTML** for user input
- **Flask**-based backend to process data and return predictions
- **Trained ML model (.pkl)** to make temperature predictions
- Support for both **manual input** and **sensor-based input**
- Visual assets included (`motor.png`, `thermo.png`)

## 🧠 Technologies Used
- Python (Flask, Pandas, scikit-learn)
- HTML, CSS
- Machine Learning (Regression models)
- Pickle for model serialization

## 📁 Files Included
- `app.py` – Flask app runner
- `EMTP.py` – Core logic
- `train_model.py` – Training script
- `best_model.pkl` – Trained model
- `Manual_predict.html`, `Sensor_predict.html`, `home.html` – Frontend pages
- `your_dataset.csv` – Dataset used
- Images and static files

## 🚀 How to Run
1. Clone the repo
2. Run `train_model.py` to train or update the model
3. Launch `app.py`
4. Open browser at `http://localhost:5000/`
5. Input values and get the predicted temperature

---

This project was built as part of an academic exercise to apply ML concepts in a real-world context.
