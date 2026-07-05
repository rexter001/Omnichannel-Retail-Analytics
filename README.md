# Omnichannel Retail Customer Behavior Analytics & Multi-Horizon Demand Forecasting System

## AI/ML Internship – Week 3 Mini Project

## Project Overview

This project develops an end-to-end machine learning system for retail customer analytics using the Online Retail II dataset. The system analyzes customer purchasing behavior, segments customers into meaningful groups, detects anomalous transactions, and forecasts future sales demand to support business decision-making.

The project combines data preprocessing, feature engineering, customer segmentation, anomaly detection, and time-series forecasting into a complete analytics pipeline.

---

# Objectives

- Clean and preprocess retail transaction data.
- Perform RFM (Recency, Frequency, Monetary) analysis.
- Engineer meaningful customer features.
- Segment customers using clustering techniques.
- Detect anomalous customer transactions.
- Forecast future sales demand using time-series models.
- Generate business insights for marketing and inventory optimization.

---

# Dataset

**Dataset:** Online Retail II Dataset

**Source:** UCI Machine Learning Repository

Dataset contains retail transaction records including:

- Invoice
- StockCode
- Description
- Quantity
- InvoiceDate
- Price
- Customer ID
- Country

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Statsmodels
- Prophet
- Google Colab
- Git & GitHub

---

# Project Workflow

```
Raw Retail Dataset
        │
        ▼
Data Cleaning & RFM Analysis
(Member 1)
        │
        ▼
Feature Engineering
(Member 2)
        │
        ▼
Customer Segmentation
(PCA + K-Means)
(Member 3)
        │
        ▼
Anomaly Detection &
Demand Forecasting
(Member 4)
        │
        ▼
Business Strategy &
Final Analytics Report
```

---

# Repository Structure

```
Omnichannel-Retail-Analytics/

│── member 1/
│     ├── data/
│     ├── cleaned_data/
│     ├── notebooks/
│     ├── scripts/
│     ├── requirements.txt
│     └── README.md

│── member 2/
│     ├── data/
│     ├── notebooks/
│     ├── scripts/
│     └── README.md

│── member 3/
│     ├── notebooks/
│     ├── data/
│     └── README.md

│── member4/
│     ├── notebooks/
│     ├── results/
│     ├── requirements.txt
│     └── README.md

│── Executive_Analytical_Report.pdf
│── Contribution_Log.pdf
│── README.md
```

---

# Machine Learning Pipeline

### Data Preprocessing
- Missing value handling
- Duplicate removal
- Data cleaning
- RFM feature generation

### Feature Engineering
- Customer-level feature creation
- Data scaling
- Pipeline generation

### Customer Segmentation
- PCA
- K-Means Clustering
- Elbow Method
- Silhouette Score Evaluation

### Anomaly Detection
- Isolation Forest
- Customer outlier detection
- Outlier visualization

### Demand Forecasting
- Daily sales aggregation
- Time Series Decomposition
- ARIMA Forecasting
- Prophet Forecasting
- Forecast Evaluation (RMSE & MAPE)

---

# Business Insights

The analytical system enables businesses to:

- Identify high-value loyal customers.
- Detect customers at risk of churn.
- Discover abnormal purchasing behavior.
- Forecast future product demand.
- Improve inventory planning.
- Design targeted marketing campaigns.

---

# Team Contributions

| Member | Responsibility |
|---------|----------------|
| Member 1 | Data Cleaning, Missing Value Handling, RFM Analysis |
| Member 2 | Feature Engineering, Data Scaling, Feature Pipeline |
| Member 3 | PCA, K-Means Clustering, Customer Segmentation |
| Member 4 | Isolation Forest, ARIMA, Prophet Forecasting, Model Evaluation |
| Member 5 (Team Lead) | GitHub Integration, Project Management, Business Insights, Final Documentation, Repository Integration |

---

# How to Run

1. Clone the repository

```bash
git clone https://github.com/rexter001/Omnichannel-Retail-Analytics.git
```

2. Install dependencies

```bash
pip install -r member4/requirements.txt
```

3. Run the notebooks in order:

- Member 1 – Data Cleaning & RFM Analysis
- Member 2 – Feature Engineering
- Member 3 – Customer Segmentation
- Member 4 – Anomaly Detection & Demand Forecasting

---

# Results

The project successfully demonstrates:

- Customer Segmentation using RFM Analysis
- PCA-based Cluster Visualization
- K-Means Clustering
- Anomaly Detection using Isolation Forest
- Time Series Decomposition
- ARIMA Sales Forecasting
- Prophet Sales Forecasting
- Forecast Evaluation Metrics
- Actionable Business Recommendations

---

# Future Enhancements

- Real-time sales forecasting
- Interactive Power BI dashboard
- Automated customer recommendation system
- Deep learning-based forecasting models
- Cloud deployment using AWS or Azure

---

# References

1. UCI Machine Learning Repository – Online Retail II Dataset
2. Scikit-learn Documentation
3. Statsmodels Documentation
4. Prophet Documentation
5. Python Documentation

---

## Team

**AI/ML Internship – Week 3 Mini Project**

**Team Lead:** Member 5

**Repository Maintained By:** Team Lead