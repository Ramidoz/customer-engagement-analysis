import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
cleaned_data_path = "data/cleaned_customer_data.csv"
df = pd.read_csv(cleaned_data_path)

# Convert date columns back to datetime
df["signup_date_time"] = pd.to_datetime(df["signup_date_time"])
df["cancel_date_time"] = pd.to_datetime(df["cancel_date_time"], errors='coerce')

# Set up plots
sns.set_style("whitegrid")

# 📌 1️⃣ **Customer Demographics**
plt.figure(figsize=(8, 5))
sns.histplot(df["age"], bins=20, kde=True)
plt.title("Age Distribution of Customers")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# 📌 2️⃣ **Subscription Distribution**
plt.figure(figsize=(6, 4))
sns.countplot(x="name", data=df, palette="viridis")
plt.title("Subscription Type Distribution")
plt.xlabel("Subscription Type")
plt.ylabel("Count")
plt.show()

# 📌 3️⃣ **Churn Analysis**
df["is_active"] = df["cancel_date_time"].isna()

plt.figure(figsize=(6, 4))
sns.countplot(x=df["is_active"].map({True: "Active", False: "Churned"}), palette="coolwarm")
plt.title("Customer Churn vs. Active Customers")
plt.xlabel("Status")
plt.ylabel("Count")
plt.show()

# 📌 4️⃣ **Support Cases Trend**
if "date_time" in df.columns:
    df["case_month"] = pd.to_datetime(df["date_time"]).dt.to_period("M")

    plt.figure(figsize=(10, 5))
    df["case_month"].value_counts().sort_index().plot(kind="line", marker="o", color="blue")
    plt.title("Customer Support Cases Over Time")
    plt.xlabel("Month")
    plt.ylabel("Number of Cases")
    plt.xticks(rotation=45)
    plt.show()

print("✅ EDA Completed: Visualizations generated successfully!")
