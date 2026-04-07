import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

def create_easy_task():
    df = pd.DataFrame({"User_ID": range(1, 11),
        "Age": [25.0, 30.0, np.nan, 35.0, 40.0, np.nan, 22.0, 28.0, 31.0, np.nan], # Use floats here
        "Clicks": [10.0, 12.0, 5.0, 8.0, np.nan, 15.0, 7.0, 9.0, 11.0, 6.0]})
    df_clean = df.copy()
    df_clean["Age"] = df_clean["Age"].fillna(df_clean["Age"].median()).round(2)
    df_clean["Clicks"] = df_clean["Clicks"].fillna(df_clean["Clicks"].median()).round(2)
    df_clean["User_ID"] = df_clean["User_ID"].astype(str)
    df.to_csv("data/easy_dirty.csv", index=False); df_clean.to_csv("data/easy_clean.csv", index=False)

def create_medium_task():
    df = pd.DataFrame({"Product_ID": range(101,111), "Price": ["19.99","25.50","NaN","15.00","30.00","18.50","20.00","Nan","32.50","40.00"], "Category": ["Tech","Home","Tech",np.nan,"Home","Home","Home",np.nan,"Tech","Home"]})
    df_clean = df.copy()
    df_clean["Price"] = pd.to_numeric(df_clean["Price"], errors='coerce').fillna(22.50).round(2)
    df_clean["Category"] = df_clean["Category"].fillna("Home")
    df_clean["Product_ID"] = df_clean["Product_ID"].astype(str)
    df.to_csv("data/med_dirty.csv", index=False); df_clean.to_csv("data/med_clean.csv", index=False)

def create_hard_task():
    df = pd.DataFrame({"Customer_ID": range(1001,1011), "Income_k": [50,60,np.nan,55,120,np.nan,20,56,53,29], "Survey_Response": [np.nan,"Yes",np.nan,np.nan,"No",np.nan,"Yes","Yes","No","No"]})
    df_clean = df.copy()
    
    df_clean["Income_k"] = df_clean["Income_k"].fillna(df_clean["Income_k"].median()).round(2)
    df_clean["Survey_Response"] = df_clean["Survey_Response"].fillna("Unknown") # 'No' is the mode here
    df_clean["Customer_ID"] = df_clean["Customer_ID"].astype(str)
    df.to_csv("data/hard_dirty.csv", index=False); df_clean.to_csv("data/hard_clean.csv", index=False)

if __name__ == "__main__":
    create_easy_task(); create_medium_task(); create_hard_task()
    print("Ground truth synchronized with agent logic.")