"""Generates synthetic messy and clean data for AutoClean-Pro tasks."""

import pandas as pd
import numpy as np
import os

# Create the data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

def create_easy_task():
    """Easy: Basic numeric NaN imputation (Mean/Median/Fillna)."""
    data = {
        "User_ID": range(1, 11),
        "Age": [25, 30, np.nan, 35, 40, np.nan, 22, 28, 31, np.nan],
        "Clicks": [10, 12, 5, 8, np.nan, 15, 7, 9, 11, 6]
    }
    df_dirty = pd.DataFrame(data)
    df_clean = df_dirty.copy()
    df_clean["Age"] = df_clean["Age"].fillna(df_clean["Age"].median())
    df_clean["Clicks"] = df_clean["Clicks"].fillna(0)
    
    df_dirty.to_csv("data/easy_dirty.csv", index=False)
    df_clean.to_csv("data/easy_clean.csv", index=False)

def create_medium_task():
    """Medium: Type casting and categorical mode imputation."""
    data = {
        "Product_ID": [101, 102, 103, 104, 105],
        "Price": ["19.99", "25.50", "NaN", "15.00", "30.00"], # Strings!
        "Category": ["Tech", "Home", "Tech", np.nan, "Home"]
    }
    df_dirty = pd.DataFrame(data)
    
    # Ground truth
    df_clean = df_dirty.copy()
    df_clean["Price"] = pd.to_numeric(df_clean["Price"], errors='coerce').fillna(20.0)
    df_clean["Category"] = df_clean["Category"].fillna("Tech")
    
    df_dirty.to_csv("data/med_dirty.csv", index=False)
    df_clean.to_csv("data/med_clean.csv", index=False)

def create_hard_task():
    """Hard: The 50% Rule (HITL) and KNN Imputation."""
    data = {
        "Customer_ID": [1001, 1002, 1003, 1004, 1005, 1006],
        "Income_k": [50, 60, np.nan, 55, 120, np.nan], # Needs KNN (5-25% missing)
        "Survey_Response": [np.nan, "Yes", np.nan, np.nan, "No", np.nan] # >50% missing!
    }
    df_dirty = pd.DataFrame(data)
    
    # Ground truth for Hard
    df_clean = df_dirty.copy()
    df_clean["Income_k"] = df_clean["Income_k"].fillna(65.0) # Simplified KNN result
    # For HITL, the target is actually just flagging, but we provide a clean state
    df_clean["Survey_Response"] = df_clean["Survey_Response"].fillna("Review Required")
    
    df_dirty.to_csv("data/hard_dirty.csv", index=False)
    df_clean.to_csv("data/hard_clean.csv", index=False)

if __name__ == "__main__":
    create_easy_task()
    create_medium_task()
    create_hard_task()
    print("Successfully generated all CSV files in /data folder!")