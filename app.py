from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import base64
import pickle

app = Flask(__name__, template_folder="web_app/templates")  # ✅ Ensuring correct templates folder

# Load Data
df = pd.read_csv("data/cleaned_customer_data.csv")

# Load ML Model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Function to Create Matplotlib Plots
def plot_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# ✅ Homepage Route
@app.route('/')
def home():
    return render_template('index.html')  # ✅ Corrected

# ✅ Visualizations Route
@app.route('/visualizations')
def visualizations():
    # Churn Plot
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='churn', hue='churn', palette="coolwarm", legend=False, ax=ax1)
    plt.title("Churned vs Active Customers")
    churn_plot = plot_to_base64(fig1)

    # Signup Trends
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    df["signup_date_time"] = pd.to_datetime(df["signup_date_time"])
    df["signup_date_time"].dt.month.value_counts().sort_index().plot(kind="bar", color="skyblue", ax=ax2)
    plt.title("Customer Signups by Month")
    signup_plot = plot_to_base64(fig2)

    return render_template('visualizations.html', churn_plot=churn_plot, signup_plot=signup_plot)  # ✅ Corrected

# ✅ Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        price = float(data["price"])
        signup_timestamp = pd.to_datetime(data["signup_date_time"]).timestamp()  # Convert to numerical format

        prediction = model.predict([[signup_timestamp, price]])[0]
        result = "Churned" if prediction == 1 else "Active"

        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

# ✅ Report Download Route
@app.route('/download_report')
def download_report():
    df.to_csv("customer_engagement_report.csv", index=False)
    return jsonify({"message": "Report saved as customer_engagement_report.csv"})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)  # ✅ Ensure it's accessible on EC2
