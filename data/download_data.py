"""
DATA DOWNLOAD SCRIPT
====================

Purpose:
    Downloads the Telco Customer Churn dataset from IBM's GitHub repository
    and saves it locally for model training.

Input:
    - None (fetches from remote URL)
    - URL: IBM Telco Customer Churn dataset (CSV format)

Output:
    - data/telco_churn.csv (7,043 rows × 21 columns)
    - Console output showing download success and churn distribution

Dataset Info:
    - customerID: Unique identifier
    - Demographics: gender, SeniorCitizen, Partner, Dependents
    - Services: PhoneService, InternetService, OnlineSecurity, etc.
    - Account: tenure, Contract, PaymentMethod, MonthlyCharges, TotalCharges
    - Target: Churn (Yes/No)

Connection to Other Files:
    → train_model.py reads the CSV this script creates
    
When to Run:
    - Once before training
    - Re-run if dataset is updated or deleted

Example Usage:
    $ python3 data/download_data.py
    ✅ Downloaded 7043 rows
    Churn: {'No': 5174, 'Yes': 1869}
"""


import pandas as pd

url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
df.to_csv('data/telco_churn.csv', index=False)
print(f"✅ Downloaded {len(df)} rows")
print(f"Churn: {df['Churn'].value_counts().to_dict()}")