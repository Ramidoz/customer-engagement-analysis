from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__, template_folder="web_app/templates")

# Load Model & Encoders
with open("models/churn_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("models/label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("models/label_encoder_subscription.pkl", "rb") as file:
    label_encoder_subscription = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None

    if request.method == "POST":
        try:
            # ✅ Step 1: Debugging - Print form data
            print("🔹 Received Input:", request.form)

            # ✅ Step 2: Extract form data
            age = int(request.form["age"])
            gender = request.form["gender"]
            subscription_type = request.form["subscription_type"]
            price = float(request.form["price"])
            billing_cycle = int(request.form["billing_cycle"])

            # ✅ Step 3: Encode categorical values
            gender_encoded = label_encoder_gender.transform([gender])[0]
            subscription_encoded = label_encoder_subscription.transform([subscription_type])[0]

            # ✅ Step 4: Make Prediction
            prediction = model.predict([[age, gender_encoded, subscription_encoded, price, billing_cycle]])[0]
            probability = model.predict_proba([[age, gender_encoded, subscription_encoded, price, billing_cycle]])[0][1]

            # ✅ Step 5: Convert prediction result
            prediction_text = "Churned" if prediction == 1 else "Active"

            print(f"🔹 Prediction: {prediction_text}, Probability: {probability:.2f}")

        except Exception as e:
            print(f"❌ Error in Prediction: {e}")
            prediction_text = "Error in prediction"
            probability = "N/A"

    return render_template("index.html", prediction=prediction_text, probability=probability)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
