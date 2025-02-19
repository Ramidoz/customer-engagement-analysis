from flask import Flask, render_template, request, jsonify
import requests
import os

app = Flask(__name__)

# ✅ Use Environment Variable for API URL
API_URL = os.getenv("API_URL", "http://3.139.84.151:5000/predict")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    customer_segment = None  # Ensure it's initialized

    if request.method == "POST":
        try:
            # ✅ Get input from form
            age = int(request.form["age"])
            gender = request.form["gender"]
            subscription_type = request.form["subscription_type"]
            price = float(request.form["price"])
            billing_cycle = int(request.form["billing_cycle"])

            # ✅ Prepare data for API
            input_data = {
                "age": age,
                "gender": gender,
                "subscription_type": subscription_type,
                "price": price,
                "billing_cycle": billing_cycle
            }

            # ✅ Call the API with Error Handling
            try:
                response = requests.post(API_URL, json=input_data, timeout=5)  # Add timeout
                response.raise_for_status()  # Raise error for bad status codes
                result = response.json()
            except requests.exceptions.RequestException as e:
                print(f"🚨 API Error: {e}")
                result = {"churn_prediction": None, "churn_probability": "N/A", "customer_segment": "API Error"}

            # ✅ Extract Results Safely
            prediction = "Churn" if result["churn_prediction"] == 1 else "Active"
            probability = result["churn_probability"]
            customer_segment = result.get("customer_segment", "Not Available")  # Use .get() to avoid KeyError

        except Exception as e:
            print(f"🚨 Form Processing Error: {e}")
            prediction = "Error"
            probability = "N/A"
            customer_segment = "Error in Form Submission"

    return render_template("index.html", prediction=prediction, probability=probability, customer_segment=customer_segment)

if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "False") == "True"
    app.run(debug=debug_mode, host="0.0.0.0", port=5001)
