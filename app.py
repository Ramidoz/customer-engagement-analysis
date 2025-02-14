import pandas as pd
import pickle
from flask import Flask, request, jsonify

# 📌 Step 1: Load the Trained Models and Encoders
app = Flask(__name__)

with open("models/churn_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("models/label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("models/label_encoder_subscription.pkl", "rb") as f:
    label_encoder_subscription = pickle.load(f)

with open("models/customer_segmentation.pkl", "rb") as f:
    kmeans = pickle.load(f)

@app.route('/')  # Homepage route
def home():
    return "✅ Flask API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.json

        # Convert input data to DataFrame
        input_data = pd.DataFrame([data])

        # Encode categorical variables
        input_data["gender_encoded"] = label_encoder_gender.transform([data["gender"]])[0]
        input_data["subscription_type_encoded"] = label_encoder_subscription.transform([data["subscription_type"]])[0]

        # Select features for churn prediction
        input_features = input_data[["age", "gender_encoded", "subscription_type_encoded", "price", "billing_cycle"]]

        # Make churn prediction
        prediction = model.predict(input_features)[0]
        probability = model.predict_proba(input_features)[0][1]  # Probability of churn

        # Normalize features for segmentation (same as training)
        segment_features = input_data[["subscription_type_encoded", "price", "billing_cycle", "age"]]

        try:
            customer_segment = kmeans.predict(segment_features)[0]
            # Map segments to labels
            segment_labels = {0: "High-Value", 1: "At-Risk"}  # Modify based on your clusters
            segment_name = segment_labels.get(customer_segment, "Unknown")
        except Exception as e:
            segment_name = "Segmentation Error"

        return jsonify({
            "churn_prediction": int(prediction),
            "churn_probability": round(probability, 2),
            "customer_segment": segment_name  # Always return a segment value
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)  # Disable debug mode for production
