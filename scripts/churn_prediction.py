import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 📌 Step 1: Load the cleaned dataset
df = pd.read_csv("data/cleaned_customer_data.csv")

# Convert dates to datetime
df["signup_date_time"] = pd.to_datetime(df["signup_date_time"])
df["cancel_date_time"] = pd.to_datetime(df["cancel_date_time"], errors='coerce')

# Create a target variable (1 = Churned, 0 = Active)
df["churned"] = df["cancel_date_time"].notna().astype(int)

# 📌 Step 2: Data Preprocessing
# Encode categorical variables
label_encoder = LabelEncoder()
df["gender_encoded"] = label_encoder.fit_transform(df["gender"])
df["subscription_type_encoded"] = label_encoder.fit_transform(df["name"])  # Subscription Type

# Select relevant features
features = ["age", "gender_encoded", "subscription_type_encoded", "price", "billing_cycle"]
target = "churned"

X = df[features]
y = df[target]

# Split dataset into training & test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Step 3: Train the Machine Learning Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📌 Step 4: Make Predictions & Evaluate Model
y_pred = model.predict(X_test)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 📌 Step 5: Visualize Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=["Active", "Churned"], yticklabels=["Active", "Churned"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("✅ Churn Prediction Model Completed!")
