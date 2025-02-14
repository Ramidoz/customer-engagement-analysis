import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# 📌 Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# 📌 Load the Cleaned Dataset
df = pd.read_csv("data/cleaned_customer_data.csv")

# Convert date columns
df["signup_date_time"] = pd.to_datetime(df["signup_date_time"])
df["cancel_date_time"] = pd.to_datetime(df["cancel_date_time"], errors='coerce')

# Create churn label (1 = Churned, 0 = Active)
df["churned"] = df["cancel_date_time"].notna().astype(int)

# 📌 Encode categorical variables
label_encoder_gender = LabelEncoder()
df["gender_encoded"] = label_encoder_gender.fit_transform(df["gender"])

label_encoder_subscription = LabelEncoder()
df["subscription_type_encoded"] = label_encoder_subscription.fit_transform(df["name"])

# 📌 Save label encoders
with open("models/label_encoder_gender.pkl", "wb") as f:
    pickle.dump(label_encoder_gender, f)
with open("models/label_encoder_subscription.pkl", "wb") as f:
    pickle.dump(label_encoder_subscription, f)

# 📌 Select features for training
features = ["age", "gender_encoded", "subscription_type_encoded", "price", "billing_cycle"]
target = "churned"

X = df[features]
y = df[target]

# 📌 Train the ML Model
model = RandomForestClassifier(n_estimators=50, random_state=42)  # Reduce estimators for efficiency
model.fit(X, y)

# 📌 Save the trained model
with open("models/churn_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

print("✅ Model training completed! Files saved in 'models/' directory.")
