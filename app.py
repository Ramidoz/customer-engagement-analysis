from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import base64
import pickle

app = Flask(__name__, template_folder="web_app/templates")

# ✅ Load Data
data_path = "data/cleaned_customer_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Data file '{data_path}' not found!")
df = pd.read_csv(data_path)

# ✅ Load ML Model from `models/`
model_path = "models/churn_model.pkl"  # ✅ Updated Path
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file '{model_path}' not found!")

with open(model_path, "rb") as file:
    model = pickle.load(file)

# ✅ Function to Convert Plots to Base64
def plot_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# ✅ Homepage Route
@app.route('/')
def home():
    return render_template('index.html')

# ✅ Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided!"}), 400

        # Validate Input
        if "price" not in data or "signup_date_time" not in data:
            return jsonify({"error": "Missing 'price' or 'signup_date_time' field!"}), 400

        price = float(data["price"])
        signup_timestamp = pd.to_datetime(data["signup_date_time"], errors='coerce')

        if pd.isnull(signup_timestamp):
            return jsonify({"error": "Invalid date format!"}), 400

        signup_timestamp = signup_timestamp.timestamp()  # Convert to numerical format

        # ✅ Make Prediction
        prediction = model.predict([[signup_timestamp, price]])[0]
        result = "Churned" if prediction == 1 else "Active"

        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
