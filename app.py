from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the saved models
dt_model = joblib.load("decision_tree_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return "Titanic Survival Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Receive input as JSON
    features = np.array(data["features"]).reshape(1, -1)

    # Get predictions
    dt_prediction = dt_model.predict(features)[0]
    rf_prediction = rf_model.predict(features)[0]
    logreg_prediction = logreg_prediction(features)[0]

    return jsonify({
        "Decision_Tree_Prediction": int(dt_prediction),
        "Random_Forest_Prediction": int(rf_prediction),
        "Logistic_Regression_Prediction": int(logreg_prediction)
    })

if __name__ == "__main__":
    app.run(debug=True)
