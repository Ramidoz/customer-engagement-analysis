from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# API endpoint for churn prediction
API_URL = "http://3.139.84.151:5000/predict"

# Load Data
df = pd.read_csv("data/cleaned_customer_data.csv")

# Function to Convert Plots to Base64
def plot_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

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

        # Call the Prediction API
        response = requests.post(API_URL, json=input_data)
        if response.status_code == 200:
            result = response.json()
            prediction = "Churn" if result["churn_prediction"] == 1 else "Active"
            probability = result["churn_probability"]

    return render_template("index.html", prediction=prediction, probability=probability)

@app.route("/dashboard")
def dashboard():
    # 📊 **Churn Rate Visualization**
    fig1, ax1 = plt.subplots(figsize=(6,4))
    sns.countplot(data=df, x="churn", hue="churn", palette="coolwarm", legend=False, ax=ax1)
    plt.title("Churned vs Active Customers")
    churn_plot = plot_to_base64(fig1)

    # 📊 **Subscription Type Breakdown**
    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.countplot(data=df, x="subscription_type", palette="viridis", ax=ax2)
    plt.title("Subscription Types Distribution")
    subscription_plot = plot_to_base64(fig2)

    # 📊 **Revenue by Billing Cycle**
    fig3, ax3 = plt.subplots(figsize=(6,4))
    df.groupby("billing_cycle")["price"].sum().plot(kind="bar", ax=ax3, color="coral")
    plt.title("Total Revenue by Billing Cycle")
    revenue_plot = plot_to_base64(fig3)

    return render_template(
        "dashboard.html",
        churn_plot=churn_plot,
        subscription_plot=subscription_plot,
        revenue_plot=revenue_plot
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
