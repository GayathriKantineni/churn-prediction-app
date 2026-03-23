# 🚀 Customer Churn Prediction System

## 📌 Overview

This project predicts whether a customer is likely to churn (leave a service) using Machine Learning. It helps businesses identify high-risk customers and take proactive actions to improve retention.

The system includes:

* Data preprocessing pipeline
* XGBoost machine learning model
* Interactive Streamlit web application
* Live deployment for real-time predictions

---

## 🎯 Problem Statement

Customer churn is a major issue for businesses. Acquiring new customers is more expensive than retaining existing ones.

This project aims to:

* Predict churn probability
* Identify key factors influencing churn
* Enable data-driven retention strategies

---

## 🧠 Solution Approach

### 🔹 1. Data Preprocessing

* Removed unnecessary columns (e.g., customerID)
* Handled missing values in `TotalCharges`
* Encoded categorical variables

### 🔹 2. Feature Engineering

* Used customer tenure, charges, and service details
* Converted categorical data into numerical format

### 🔹 3. Model Building

* Implemented **XGBoost Classifier**
* Compared with baseline models
* Optimized using hyperparameters

### 🔹 4. Model Evaluation

* ROC-AUC Score: ~0.85+
* Evaluated using Precision, Recall, F1-score

### 🔹 5. Deployment

* Built a web app using Streamlit
* Enabled real-time churn prediction

---

## 📊 Key Insights

* Customers with **low tenure** are more likely to churn
* **Month-to-month contracts** have higher churn rates
* High **monthly charges** increase churn probability
* Customers using more services tend to stay longer

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Matplotlib, Seaborn

---

## 🌐 Live Demo

👉 

---

## 📂 Project Structure

```bash
churn_project/
│
├── data/
│   └── churn.csv
│
├── models/
│   ├── churn_model.pkl
│   └── scaler.pkl
│
├── train.py
├── app.py
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run Locally

### 1. Clone Repository

```bash
git clone https://github.com/your-username/churn-prediction-app.git
cd churn-prediction-app
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Model

```bash
python train.py
```

### 4. Run App

```bash
streamlit run app.py
```

---

## 💡 Business Impact

* Identifies high-risk customers early
* Enables targeted retention campaigns
* Helps reduce churn and increase revenue

---

## 🚀 Future Improvements

* Use full feature set in Streamlit app
* Add real-time data integration
* Improve UI with dashboards and analytics
* Deploy using cloud platforms (AWS/GCP)

---

## 👩‍💻 Author

**Gayathri**

* GitHub:https://github.com/GayathriKantineni
* LinkedIn:www.linkedin.com/in/kantineni-gayathri-686267289

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!
