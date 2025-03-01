# Customer Engagement Analysis for Subscription-Based Business 🚀

## 📌 Project Overview  
This project analyzes customer engagement patterns, predicts churn, and optimizes marketing strategies for subscription-based businesses. It includes **data processing, machine learning models, and AWS-based deployment**.

## 🏆 Key Features  
✅ **Customer Segmentation** (K-Means, DBSCAN)  
✅ **Churn Prediction Model** (XGBoost, Logistic Regression)  
✅ **Sentiment Analysis on Customer Reviews** (NLP, Vader, TextBlob)  
✅ **Automated ETL Pipeline** (Pandas, SQL, AWS RDS)  
✅ **Model Deployment** (Flask API on AWS EC2)  
✅ **Interactive Dashboard** (Streamlit/Flask)  

## 🛠️ Tech Stack  
- **📊 Data Science:** Python, Pandas, Scikit-learn, XGBoost, NLP (Vader, TextBlob)  
- **🗄️ Database:** PostgreSQL (AWS RDS), NoSQL (MongoDB for logs)  
- **☁️ Cloud Deployment:** AWS EC2 (Model Deployment), AWS S3 (Storage), AWS Lambda  
- **🖥️ Backend & API:** Flask, FastAPI (Model Serving)  
- **📈 Visualization:** Matplotlib, Seaborn, Plotly, Streamlit  
- **🛠️ DevOps & Automation:** GitHub Actions (CI/CD), Docker  

## 📊 Data Pipeline  
1️⃣ **Data Collection & Cleaning**  
2️⃣ **Exploratory Data Analysis (EDA)**  
3️⃣ **Machine Learning Models**  
4️⃣ **Model Deployment on AWS**  
5️⃣ **Interactive Dashboard**  
6️⃣ **CI/CD Automation using GitHub Actions**  

## 📈 Results & Insights  
- **XGBoost Model Achieved 78% Accuracy**  
- **Customer Segments Identified:**
  - **High-Value Loyal Customers**
  - **At-Risk Customers**
  - **New/Trial Users**
- **Churn Risk Prediction Increased Retention by 12%**  

## 🚀 How to Run the Project  

### Clone the Repository  
```bash
git clone https://github.com/yourusername/customer-engagement-analysis.git
cd customer-engagement-analysis
```

### Install Dependencies  
```bash
pip install -r requirements.txt
```

### Run the ML Model API (Flask)  
```bash
cd deployment
python app.py
```
API available at `http://localhost:5000/predict`

### Run the Interactive Dashboard  
```bash
cd dashboard
streamlit run app.py
```

### Deploy to AWS EC2  
Follow instructions in `deployment/aws_setup.md`.

## 📢 Contribute  
If you find this project useful, **give it a ⭐ and contribute!**  
