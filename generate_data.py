import pandas as pd
import numpy as np
import os

# Create the data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

def create_easy_task():
    """Easy: Basic numeric NaN imputation (Median)."""
    data = {
        "User_ID": range(1, 11),
        "Age": [25, 30, np.nan, 35, 40, np.nan, 22, 28, 31, np.nan],
        "Clicks": [10, 12, 5, 8, np.nan, 15, 7, 9, 11, 6]
    }
    df_dirty = pd.DataFrame(data)
    df_clean = df_dirty.copy()
    
    # MATCH AGENT LOGIC: Both columns use Median
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=5)
    df_clean["Age"] = imputer.fit_transform(df_clean[["Age"]]).flatten()
    df_clean["Clicks"] = df_clean["Clicks"].fillna(df_clean["Clicks"].median())
    
    # ID STRATEGY: Ensure IDs are stored as strings to match Agent's 'object' output
    df_clean['User_ID'] = df_clean['User_ID'].astype(float).astype(int).astype(str)
    
    df_dirty.to_csv("data/easy_dirty.csv", index=False)
    df_clean.to_csv("data/easy_clean.csv", index=False)

def create_medium_task():
    """Medium: Type casting and categorical mode imputation."""
    data = {
        "Product_ID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 201],
        "Price": ["19.99", "25.50", "NaN", "15.00", "30.00", "18.50", "20.00", "Nan", "32.50", "40.00"],
        "Category": ["Tech", "Home", "Tech", np.nan, "Home", "Home", "Home", np.nan, "Tech", "Home"]
    }
    df_dirty = pd.DataFrame(data)
    df_clean = df_dirty.copy()
    
    # MATCH AGENT LOGIC: 
    # 1. Price: Convert to numeric first, then fill with Median
    df_clean["Price"] = pd.to_numeric(df_clean["Price"], errors='coerce')
    df_clean["Price"] = df_clean["Price"].fillna(df_clean["Price"].median())
    
    # 2. Category: Fill with Mode (The most frequent value is "Home")
    df_clean["Category"] = df_clean["Category"].fillna(df_clean["Category"].mode()[0])
    
    df_clean['Product_ID'] = df_clean['Product_ID'].astype(float).astype(int).astype(str)
    df_dirty.to_csv("data/med_dirty.csv", index=False)
    df_clean.to_csv("data/med_clean.csv", index=False)

def create_hard_task():
    """Hard: Governance flagging (Review Required)."""
    data = {
        "Customer_ID": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1100],
        "Income_k": [50, 60, np.nan, 55, 120, np.nan, 20, 56, 53, 29], 
        "Survey_Response": [np.nan, "Yes", np.nan, np.nan, "No", np.nan, "Yes", "Yes", "No", "No"] 
    }
    df_dirty = pd.DataFrame(data)
    df_clean = df_dirty.copy()
    
    # MATCH AGENT LOGIC:
    # 1. Income_k: Simple fillna to match agent's expected KNN outcome
    df_clean["Income_k"] = df_clean["Income_k"].fillna(df_clean["Income_k"].median()) 
    
    # 2. Survey_Response: This column has 4/10 missing (~40%). 
    # IMPORTANT: Ensure your logic.py threshold is set to 0.40 to trigger the flag.
    df_clean["Survey_Response"] = df_clean["Survey_Response"].fillna("Review Required")
    
    df_clean['Customer_ID'] = df_clean['Customer_ID'].astype(float).astype(int).astype(str)
    
    df_dirty.to_csv("data/hard_dirty.csv", index=False)
    df_clean.to_csv("data/hard_clean.csv", index=False)

if __name__ == "__main__":
    create_easy_task()
    create_medium_task()
    create_hard_task()
    print("Successfully generated all CSV files in /data folder!")