import pandas as pd
import pickle
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# ✅ Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# 📌 Load the Cleaned Dataset
df = pd.read_csv("data/cleaned_customer_data.csv")

# ✅ Convert date columns
df["signup_date_time"] = pd.to_datetime(df["signup_date_time"])
df["cancel_date_time"] = pd.to_datetime(df["cancel_date_time"], errors='coerce')

# ✅ Create churn label (1 = Churned, 0 = Active)
df["churned"] = df["cancel_date_time"].notna().astype(int)

# 📌 Encode categorical variables
label_encoder_gender = LabelEncoder()
df["gender_encoded"] = label_encoder_gender.fit_transform(df["gender"])

label_encoder_subscription = LabelEncoder()
df["subscription_type_encoded"] = label_encoder_subscription.fit_transform(df["subscription_type"])

# 📌 Save label encoders
with open("models/label_encoder_gender.pkl", "wb") as f:
    pickle.dump(label_encoder_gender, f)
with open("models/label_encoder_subscription.pkl", "wb") as f:
    pickle.dump(label_encoder_subscription, f)

# 📌 Select features for churn prediction
features = ["age", "gender_encoded", "subscription_type_encoded", "price", "billing_cycle"]
target = "churned"

X = df[features]
y = df[target]

# 📌 Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 📌 Save the trained scaler (Fix)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# 📌 Train the XGBoost Model
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    eval_metric="logloss"
)
model.fit(X_train_scaled, y_train)

# 📌 Evaluate the Model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Churn Model Accuracy: {accuracy:.2f}")

# 📌 Save the trained churn prediction model
with open("models/churn_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# ✅ User Segmentation with K-Means Clustering
X_scaled = scaler.transform(df[["subscription_type_encoded", "price", "billing_cycle", "age"]])

# ✅ Dynamically determine number of clusters
n_clusters = min(3, len(X_scaled))  # ✅ Prevents errors when data is small

# 📌 Train K-Means Model
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
df["customer_segment"] = kmeans.fit_predict(X_scaled)

# 📌 Save the trained K-Means model
with open("models/customer_segmentation.pkl", "wb") as model_file:
    pickle.dump(kmeans, model_file)

print("✅ Model training completed! Files saved in 'models/' directory.")

