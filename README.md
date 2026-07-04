# Retail Analytics — Data Engineering & RFM Module

> AI/ML Internship Industry Mini Project  
> Module: Data Engineering + RFM Customer Segmentation

---

## Project Overview

This project implements a complete **Data Engineering and RFM (Recency, Frequency, Monetary) analysis** pipeline on the **Online Retail II** dataset.

The pipeline covers:
- Loading and profiling the raw dataset
- Full data cleaning (duplicates, cancellations, nulls, invalid values)
- Feature engineering (`TotalAmount`)
- Customer-level aggregation
- RFM metric computation using industry-standard methodology
- Saving production-ready outputs

---

## Folder Structure

```
Retail_Preprocessing/
├── data/
│   └── online_retail_II.csv          # Raw source dataset (not committed to Git)
│
├── notebooks/
│   └── Data_Preprocessing.ipynb      # End-to-end Jupyter notebook
│
├── scripts/
│   ├── load_data.py                  # Task 1 – Load & profile raw data
│   ├── clean_data.py                 # Task 2 – Full cleaning pipeline
│   └── rfm_analysis.py              # Task 3 – Customer RFM aggregation
│
├── cleaned_data/
│   ├── cleaned_retail.csv            # Output: cleaned dataset
│   └── customer_rfm.csv             # Output: customer RFM table
│
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## Dataset

| Property    | Value                                          |
|-------------|------------------------------------------------|
| Name        | Online Retail II                               |
| Source      | UCI Machine Learning Repository               |
| Period      | Dec 2009 – Dec 2011                            |
| Records     | ~1,067,371 rows (before cleaning)             |
| Columns     | Invoice, StockCode, Description, Quantity,    |
|             | InvoiceDate, Price, Customer ID, Country      |

> Place the CSV at `data/online_retail_II.csv` before running the pipeline.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Retail_Preprocessing.git
cd Retail_Preprocessing
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

### Option A — Run Python scripts individually

```bash
# Step 1: Load and profile the raw dataset
python scripts/load_data.py

# Step 2: Clean the dataset and save cleaned_retail.csv
python scripts/clean_data.py

# Step 3: Build the RFM table and save customer_rfm.csv
python scripts/rfm_analysis.py
```

### Option B — Run the Jupyter Notebook

```bash
jupyter notebook notebooks/Data_Preprocessing.ipynb
```

Execute cells top-to-bottom. The notebook documents every step with markdown explanations and runs the full pipeline end-to-end.

---

## Output Files

| File                              | Description                                      |
|-----------------------------------|--------------------------------------------------|
| `cleaned_data/cleaned_retail.csv` | Cleaned dataset ready for downstream ML tasks    |
| `cleaned_data/customer_rfm.csv`   | Customer-level RFM metrics (one row per customer)|

### cleaned_retail.csv schema

| Column       | Type        | Description                            |
|--------------|-------------|----------------------------------------|
| Invoice      | str         | Invoice number                         |
| StockCode    | str         | Product/stock code                     |
| Description  | str         | Product description                    |
| Quantity     | int         | Units purchased (positive only)        |
| InvoiceDate  | datetime    | Date and time of transaction           |
| Price        | float       | Unit price in GBP                      |
| Customer ID  | int64       | Unique customer identifier             |
| Country      | str         | Customer's country                     |
| TotalAmount  | float       | Quantity × Price (line-item revenue)   |

### customer_rfm.csv schema

| Column      | Type    | Description                                        |
|-------------|---------|-----------------------------------------------------|
| Customer ID | int64   | Unique customer identifier                          |
| Recency     | int     | Days since last purchase (lower = more recent)      |
| Frequency   | int     | Number of unique invoices (higher = more frequent)  |
| Monetary    | float   | Total spend in GBP (higher = more valuable)         |

---

## Data Cleaning Steps

| Step | Action                               | Reason                                      |
|------|--------------------------------------|---------------------------------------------|
| 1    | Remove duplicate rows                | Avoid double-counting transactions          |
| 2    | Remove cancelled invoices (C prefix) | Credits/returns inflate cancellation rates  |
| 3    | Drop missing Customer ID             | Cannot perform customer-level analysis      |
| 4    | Drop missing Description             | Invalid product records                     |
| 5    | Remove Quantity ≤ 0                  | Returns/errors distort revenue              |
| 6    | Remove Price ≤ 0                     | Invalid pricing records                     |
| 7    | Parse InvoiceDate to datetime        | Enables time-series and recency calcs       |
| 8    | Create TotalAmount feature           | Enables monetary aggregation                |

---

## RFM Methodology

```
Reference Date = max(InvoiceDate) + 1 day

Recency   = (Reference Date − Last Purchase Date).days   [per customer]
Frequency = COUNT DISTINCT Invoice                        [per customer]
Monetary  = SUM(TotalAmount)                              [per customer]
```

The +1-day convention ensures the most-recent customer has Recency ≥ 1 and
avoids data leakage from using an arbitrary future 
