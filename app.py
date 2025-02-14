from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import base64
import pickle

app = Flask(__name__, template_folder="web_app/templates")  # ✅ Ensuring correct templates folder

# ✅ Check if Data File Exists
data_path = "data/cleaned_customer_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Data file '{data_path}' not found!")

df = pd.read_csv(data_path)

# ✅ Check if Model File Exists
model_path = "model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file '{model_path}' not found!")

# Load ML Model
with open(model_path, "rb") as file:
    model = pickle.load(file)


# ✅ Function to Convert Plots to Base64 (For Display in HTML)
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
    try:
        # 📊 Churn Distribution
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='churn', hue='churn', palette="coolwarm", legend=False, ax=ax1)
        plt.title("Churned vs Active Customers")
        churn_plot = plot_to_base64(fig1)

        # 📊 Customer Signups Trends
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        df["signup_date_time"] = pd.to_datetime(df["signup_date_time"], errors='coerce')
        df["signup_date_time"].dt.month.value_counts().sort_index().plot(kind="bar", color="skyblue", ax=ax2)
        plt.title("Customer Signups by Month")
        signup_plot = plot_to_base64(fig2)

        return render_template('visualizations.html', churn_plot=churn_plot, signup_plot=signup_plot)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided!"}), 400

        # ✅ Validate Input
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


# ✅ Report Download Route
@app.route('/download_report')
def download_report():
    try:
        report_path = "customer_engagement_report.csv"
        df.to_csv(report_path, index=False)
        return jsonify({"message": f"✅ Report saved as '{report_path}'"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)  # ✅ Ensure it's accessible on EC2
