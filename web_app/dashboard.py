from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# API endpoint (Replace with your actual EC2 IP)
API_URL = "http://3.139.84.151:5000/predict"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None

    if request.method == "POST":
        # Get input from form
        age = int(request.form["age"])
        gender = request.form["gender"]
        subscription_type = request.form["subscription_type"]
        price = float(request.form["price"])
        billing_cycle = int(request.form["billing_cycle"])

        # Prepare data for API
        input_data = {
            "age": age,
            "gender": gender,
            "subscription_type": subscription_type,
            "price": price,
            "billing_cycle": billing_cycle
        }

        # Call the API
        response = requests.post(API_URL, json=input_data)
        if response.status_code == 200:
            result = response.json()
            prediction = "Churn" if result["churn_prediction"] == 1 else "Active"
            probability = result["churn_probability"]

    return render_template("index.html", prediction=prediction, probability=probability)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
