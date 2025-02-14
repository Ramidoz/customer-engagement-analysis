import pandas as pd
import pickle
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

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

# 📌 Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Train the XGBoost Model
model = XGBClassifier(
    n_estimators=100,        # Number of trees
    learning_rate=0.05,      # Step size shrinkage
    max_depth=4,             # Maximum depth of trees
    random_state=42,
    use_label_encoder=False, # Avoid warnings
    eval_metric="logloss"    # Recommended for classification
)
model.fit(X_train, y_train)

# 📌 Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")

# 📌 Save the trained model
with open("models/churn_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

print("✅ Model training completed! Files saved in 'models/' directory.")
