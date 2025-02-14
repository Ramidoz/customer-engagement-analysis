import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load datasets
customer_info = pd.read_csv("data/customer_info.csv")
customer_cases = pd.read_csv("data/customer_cases.csv")
customer_product = pd.read_csv("data/customer_product.csv")

# Convert date columns to datetime
customer_cases['date_time'] = pd.to_datetime(customer_cases['date_time'])
customer_product['signup_date_time'] = pd.to_datetime(customer_product['signup_date_time'])
customer_product['cancel_date_time'] = pd.to_datetime(customer_product['cancel_date_time'])

# Add Churn Column
customer_product['churned'] = customer_product['cancel_date_time'].notna()

# Set save directory for plots
save_dir = "visualizations/"
import os
os.makedirs(save_dir, exist_ok=True)

# 📊 **1. Customer Churn Distribution**
plt.figure(figsize=(8, 5))
sns.countplot(data=customer_product, x='churned', hue='churned', palette="coolwarm", legend=False)
plt.title("Customer Churn Distribution")
plt.xlabel("Churned (1 = Yes, 0 = No)")
plt.ylabel("Number of Customers")
plt.xticks([0, 1], ["Active", "Churned"])
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig(f"{save_dir}churn_distribution.png")
plt.show()

# 📈 **2. Monthly Churn Trends**
customer_product['cancel_month'] = customer_product['cancel_date_time'].dt.to_period('M')
monthly_churn = customer_product['cancel_month'].value_counts().sort_index()

plt.figure(figsize=(10, 5))
monthly_churn.plot(kind='line', marker='o', color='red')
plt.title("Monthly Churn Trends")
plt.xlabel("Month")
plt.ylabel("Number of Churned Customers")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig(f"{save_dir}monthly_churn_trends.png")
plt.show()

# 📞 **3. Support Case Distribution by Channel**
plt.figure(figsize=(8, 5))
sns.countplot(data=customer_cases, x='channel', hue='channel', order=customer_cases['channel'].value_counts().index, palette="viridis", legend=False)
plt.title("Customer Support Cases by Channel")
plt.xlabel("Support Channel")
plt.ylabel("Number of Cases")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig(f"{save_dir}support_case_distribution.png")
plt.show()

# ⏳ **4. Engagement Trends: Signup vs Churn Times**
fig = px.histogram(customer_product, x='signup_date_time', nbins=30, title="Customer Signups Over Time")
fig.write_html(f"{save_dir}customer_signup_trends.html")
fig.show()

fig = px.histogram(customer_product, x='cancel_date_time', nbins=30, title="Customer Churn Over Time", color_discrete_sequence=['red'])
fig.write_html(f"{save_dir}customer_churn_trends.html")
fig.show()

print("✅ Visualizations saved to 'visualizations/' folder.")

