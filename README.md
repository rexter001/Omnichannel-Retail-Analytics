# 🚀 Omnichannel Retail Customer Behavior Analytics & Multi-Horizon Demand Forecasting System

## AI/ML Internship – Week 3 Mini Project

---

## 📌 Project Overview

This project is an end-to-end **Retail Analytics and Machine Learning system** designed to extract business insights from large-scale transactional data.

It focuses on:
- Understanding customer purchasing behavior
- Segmenting customers using clustering techniques
- Detecting anomalous transactions
- Forecasting future sales demand
- Converting ML outputs into business strategies

The system combines **unsupervised learning, anomaly detection, and time series forecasting** to support data-driven decision-making in retail business operations.

---

## 👥 Team Information

- Member 1  
- Member 2  
- Member 3  
- Member 4  
- Member 5 (Team Lead)

---

## 📊 Dataset Information

**Dataset Name:** Online Retail II Dataset  
**Source:** Kaggle  


### 📌 Dataset Overview:
- Total Records: 1,067,371  
- Total Features: 8  

### 📌 Features:
- Invoice
- StockCode
- Description
- Quantity
- InvoiceDate
- Price
- Customer ID
- Country

---

## ⚙️ Project Pipeline

### 1. Import Libraries
- Pandas, NumPy for data processing  
- Matplotlib, Seaborn for visualization  
- Scikit-learn for ML models  
- Statsmodels & Prophet for forecasting  

---

### 2. Dataset Loading
- Load Kaggle dataset using Pandas  
- Convert InvoiceDate into datetime format  
- Handle missing values and inconsistencies  

---

### 3. Exploratory Data Analysis (EDA)
- Sales distribution analysis  
- Top-selling products  
- Customer purchase behavior  
- Time-based trends  

---

### 4. Data Cleaning & RFM Analysis
- Remove duplicates and null values  
- RFM (Recency, Frequency, Monetary) calculation  
- Normalize customer features  

---

### 5. Feature Engineering & Pipeline
- Aggregated customer-level features  
- StandardScaler normalization  
- Pipeline creation for ML models  

---

### 6. Customer Segmentation
**Techniques Used:**
- PCA (Dimensionality Reduction)
- K-Means Clustering

### 📌 Business Personas:
- High-Value Loyal Customers
- VIP Elite Customers
- Occasional Buyers
- Churn-Risk Customers
- Emerging Customers

---

### 7. Anomaly Detection
**Model Used:** Isolation Forest  

### 📌 Purpose:
- Detect abnormal transactions
- Identify possible fraud or unusual buying behavior

### 📌 Output:
- Outlier scatter plot
- Anomaly summary statistics

---

### 8. Time Series Forecasting

**Models Used:**
- ARIMA
- Facebook Prophet

### 📌 Components:
- Trend Analysis
- Seasonality Detection
- Residual Noise

### 📌 Evaluation Metrics:
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

---

### 9. Business Strategy & Insights

### 🎯 Customer-Based Strategies:
- High-Value customers → Loyalty rewards & premium offers  
- VIP customers → Exclusive access & retention programs  
- Occasional buyers → Discount campaigns  
- Churn-risk customers → Win-back strategies  

### 📦 Inventory Optimization:
- Use forecasting results for stock planning  
- Reduce overstock and stockout situations  
- Improve warehouse efficiency  

### 📈 Marketing Optimization:
- Personalized recommendations  
- Targeted promotions  
- Customer segmentation-based campaigns  

---

### 10. Conclusion

This project demonstrates how machine learning and time-series forecasting can transform raw retail transaction data into meaningful business insights.

It enables:
- Better customer understanding
- Improved demand forecasting
- Smarter inventory planning
- Data-driven marketing decisions

---

## 📊 Visual Deliverables Included

- 2D Cluster Scatter Plot (PCA Visualization)
- Anomaly Detection Outlier Map
- Time Series Decomposition (Trend, Seasonality, Residual)
- Model Validation Curves (ARIMA vs Prophet)

---

## 🛠️ Tools & Technologies

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Statsmodels  
- Facebook Prophet  
- Jupyter Notebook / Google Colab  

---

## 📁 Repository Structure
Omnichannel-Retail-Analytics/
│
├── member 1/
├── member 2/
├── member 3/
├── member4/
├── member5/
├── .gitignore
└── README.md   ← (FINAL ROOT README)


---

## 📚 References

- Kaggle Dataset: Online Retail II  
- Scikit-learn Documentation  
- Statsmodels Documentation  
- Prophet Documentation  
- Python Official Documentation  

---

## 🏁 Final Note

This system provides a complete end-to-end pipeline for retail analytics, combining machine learning, forecasting, and business intelligence for real-world decision-making.