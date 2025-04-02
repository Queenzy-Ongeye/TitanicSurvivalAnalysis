from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the saved models
rf_model = joblib.load("random_forest_model.pkl")
logreg_model = joblib.load("logistic_reg_model.pkl")
dt_model = joblib.load("decision_tree_model.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return "Titanic Survival Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validate JSON input
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in JSON request"}), 400

        features = np.array(data["features"]).reshape(1, -1)

        # Ensure correct number of features
        expected_features = rf_model.n_features_in_
        if len(data["features"]) != expected_features:
            return jsonify({"error": f"Expected {expected_features} features, but got {len(data['features'])}"}), 400

        # Get predictions
        rf_prediction = rf_model.predict(features)[0]
        logreg_prediction = logreg_model.predict(features)[0]
        dt_prediction = dt_model.predict(features)[0]

        return jsonify({
            "Random_Forest_Prediction": int(rf_prediction),
            "Logistic_Regression_Prediction": int(logreg_prediction),
            "Decision_tree_prediction" : int(dt_prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
