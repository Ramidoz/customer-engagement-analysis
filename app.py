import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained model and encoders
with open("models/churn_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("models/label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)
with open("models/label_encoder_subscription.pkl", "rb") as f:
    label_encoder_subscription = pickle.load(f)

@app.route('/')
def index():
    """ Serve the main dashboard """
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        input_data = pd.DataFrame([data])

        # Encode categorical variables
        input_data["gender_encoded"] = label_encoder_gender.transform([data["gender"]])[0]
        input_data["subscription_type_encoded"] = label_encoder_subscription.transform([data["subscription_type"]])[0]

        # Select required features
        input_features = input_data[["age", "gender_encoded", "subscription_type_encoded", "price", "billing_cycle"]]

        # Make prediction
        prediction = model.predict(input_features)[0]
        probability = model.predict_proba(input_features)[0][1]

        return jsonify({"churn_prediction": int(prediction), "churn_probability": round(probability, 2)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
