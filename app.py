import pandas as pd
import pickle
import logging
import os
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler

# ✅ Set up logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# ✅ Ensure all required models exist
required_files = [
    "models/churn_model.pkl",
    "models/label_encoder_gender.pkl",
    "models/label_encoder_subscription.pkl",
    "models/scaler.pkl",
    "models/customer_segmentation.pkl"
]

for file in required_files:
    if not os.path.exists(file):
        logging.error(f"❌ Missing required file: {file}")
        exit(1)  # Stop execution if a model file is missing

# ✅ Load models
with open("models/churn_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("models/label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("models/label_encoder_subscription.pkl", "rb") as f:
    label_encoder_subscription = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

try:
    with open("models/customer_segmentation.pkl", "rb") as f:
        kmeans = pickle.load(f)
    logging.info("✅ K-Means model loaded successfully!")
except Exception as e:
    logging.error(f"🚨 Error loading K-Means model: {e}")

@app.route('/')
def home():
    return "✅ Flask API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        logging.info(f"🔍 Received data: {data}")

        # Convert to DataFrame
        input_data = pd.DataFrame([data])

        # Encode categorical variables
        input_data["gender_encoded"] = label_encoder_gender.transform([data["gender"]])[0]
        input_data["subscription_type_encoded"] = label_encoder_subscription.transform([data["subscription_type"]])[0]

        # Select features for churn prediction
        input_features = input_data[["age", "gender_encoded", "subscription_type_encoded", "price", "billing_cycle"]]

        # Make churn prediction
        prediction = model.predict(input_features)[0]
        probability = model.predict_proba(input_features)[0][1]

        # Normalize features for segmentation
        segment_features = input_data[["subscription_type_encoded", "price", "billing_cycle", "age"]]
        segment_features_scaled = scaler.transform(segment_features)

        try:
            customer_segment = kmeans.predict(segment_features_scaled)[0]
            segment_labels = {0: "High-Value", 1: "At-Risk"}
            segment_name = segment_labels.get(customer_segment, "Unknown")
        except Exception as e:
            segment_name = "Segmentation Error"
            logging.error(f"🚨 Segmentation Error: {e}")

        # API Response
        response = {
            "churn_prediction": int(prediction),
            "churn_probability": round(probability, 2),
            "customer_segment": segment_name
        }
        logging.info(f"🔍 API Response: {response}")

        return jsonify(response)

    except Exception as e:
        logging.error(f"🚨 Error in API: {e}")
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
