import pandas as pd

# Define file paths
product_info_path = "data/product_info.csv"
customer_cases_path = "data/customer_cases.csv"
customer_product_path = "data/customer_product.csv"
customer_info_path = "data/customer_info.csv"

# Load datasets
df_product_info = pd.read_csv(product_info_path)
df_customer_cases = pd.read_csv(customer_cases_path)
df_customer_product = pd.read_csv(customer_product_path)
df_customer_info = pd.read_csv(customer_info_path)

# Convert date columns to datetime format (handling errors gracefully)
df_customer_cases["date_time"] = pd.to_datetime(df_customer_cases["date_time"], errors='coerce')
df_customer_product["signup_date_time"] = pd.to_datetime(df_customer_product["signup_date_time"], errors='coerce')
df_customer_product["cancel_date_time"] = pd.to_datetime(df_customer_product["cancel_date_time"], errors='coerce')

# Handle missing values in cancel_date_time (Active users haven't canceled)
df_customer_product = df_customer_product.copy()
df_customer_product["cancel_date_time"].fillna(pd.NaT, inplace=True)  # Keeps datetime consistency

# Check for duplicate entries in customer cases
df_customer_cases = df_customer_cases.drop_duplicates()

# Remove unnecessary columns (e.g., Unnamed index columns)
drop_columns = ["Unnamed: 0"]
df_customer_cases.drop(columns=drop_columns, errors="ignore", inplace=True)
df_customer_product.drop(columns=drop_columns, errors="ignore", inplace=True)
df_customer_info.drop(columns=drop_columns, errors="ignore", inplace=True)

# Merge all datasets into a single DataFrame
df_merged = df_customer_product.merge(df_customer_info, on="customer_id", how="left")
df_merged = df_merged.merge(df_product_info, left_on="product", right_on="product_id", how="left")
df_merged = df_merged.merge(df_customer_cases, on="customer_id", how="left")

# Save cleaned dataset for further analysis
output_path = "data/cleaned_customer_data.csv"
df_merged.to_csv(output_path, index=False)

print(f"✅ Data cleaning complete! Cleaned data saved as '{output_path}'")
