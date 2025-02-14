import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 📌 Step 1: Load the Cleaned Dataset
df = pd.read_csv("data/cleaned_customer_data.csv")

# Convert date columns
df["signup_date_time"] = pd.to_datetime(df["signup_date_time"])
df["cancel_date_time"] = pd.to_datetime(df["cancel_date_time"], errors='coerce')

# Create churn label (1 = Churned, 0 = Active)
df["churned"] = df["cancel_date_time"].notna().astype(int)

# Encode categorical variables
label_encoder_gender = LabelEncoder()
df["gender_encoded"] = label_encoder_gender.fit_transform(df["gender"])

label_encoder_subscription = LabelEncoder()
df["subscription_type_encoded"] = label_encoder_subscription.fit_transform(df["name"])

# Select relevant features
features = ["age", "gender_encoded", "subscription_type_encoded", "price", "billing_cycle"]
target = "churned"

X = df[features]
y = df[target]

# 📌 Step 2: Train the ML Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
with open("models/churn_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# 📌 Step 3: Create Flask App
app = Flask(__name__)

@app.route('/')  # This creates a homepage route
def home():
    return "Hello, Flask is running!"

# Load the trained model when the API starts
with open("models/churn_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.json

        # Convert input data to DataFrame
        input_data = pd.DataFrame([data])

        # Encode categorical variables using the saved encoders
        input_data["gender_encoded"] = label_encoder_gender.transform([data["gender"]])[0]
        input_data["subscription_type_encoded"] = label_encoder_subscription.transform([data["subscription_type"]])[0]

        # Select only required columns
        input_features = input_data[["age", "gender_encoded", "subscription_type_encoded", "price", "billing_cycle"]]

        # Make prediction
        prediction = model.predict(input_features)[0]
        probability = model.predict_proba(input_features)[0][1]  # Probability of churn

        # Return result as JSON
        return jsonify({"churn_prediction": int(prediction), "churn_probability": round(probability, 2)})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the API
if __name__ == "__main__":
    app.run(debug=True)
