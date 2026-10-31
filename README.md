# Multi-Company Customer Churn Prediction System

## 1. Project Overview
This Machine Learning web application predicts whether a customer is likely to cancel their subscription or stop using a service (churn) based on historical usage, spending, and support interaction data. It supports multiple industries (OTT Streaming, Telecom, and E-Commerce) and generates automated retention strategies.

## 2. System Architecture & Tech Stack
- **Frontend Interface:** Streamlit (`st.session_state` for stateful in-memory data persistence)
- **Data Processing:** Pandas & NumPy
- **Machine Learning Pipeline:** Scikit-Learn (`StandardScaler`, `OneHotEncoder`, `RandomForestClassifier`, and `LogisticRegression`)
- **Evaluation & Visualization:** Matplotlib & Seaborn (ROC-AUC Curve, Confusion Matrix, and Feature Importance)

## 3. Key Modules
1. **Customer Data Manager:** Add new customer records manually with instant AI churn prediction or bulk-generate synthetic industry datasets.
2. **Model Performance & Analytics:** Live model training with stratified 75-25 train-test split, Accuracy, ROC-AUC score, and interactive analytics charts.
3. **Churn Risk Report:** Ranks all customers into High, Medium, and Low risk tiers with actionable retention offers and one-click CSV export.

## 4. How to Run Locally
1. Install dependencies:
   pip install -r requirements.txt
2. Launch the web application:
   python -m streamlit run app.py
