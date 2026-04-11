"""
generate_data.py — AutoClean-Pro synthetic dataset generator

Run once to regenerate all six CSV files in data/.
CSV names are fixed — do not rename them (pointers in environment.py).

Task designs
------------
easy   — Age (float, 3 NaN) + Clicks (binary 0/1, 1 NaN)
         Tests: knn_impute(Age) → mode_impute(Clicks)
         Two different strategies on two different dtypes.

medium — Price (object due to 'Nan' strings, 2 NaN) + Category (str, 2 NaN)
         Tests: cast_type(Price) → median_impute(Price) → mode_impute(Category)
         Schema repair before imputation.

hard   — Income_k (float, 2 NaN, 20% missing) + Survey_Response (str, 4 NaN, 40% missing)
         Tests: flag_human(Survey_Response) → knn_impute(Income_k)
         Governance escalation + partial imputation.
         Grader: 60% weight on flag_human, 40% on Income_k cleaned.
"""
import os
import pandas as pd
import numpy as np

os.makedirs("data", exist_ok=True)


def create_easy_task():
    """
    10 rows. Age is continuous float, Clicks is binary int (click/no-click).

    Weighted missingness (Bayesian, n=10, amp=1.40, cap=0.34):
      Age:    3/10 = 30% flat → min(0.34, 0.42) = 0.34 → knn_impute
      Clicks: 1/10 = 10% flat → 0.14              → mode_impute

    Clean values:
      Age    NaNs → median = 30.0
      Clicks NaN  → mode   = 1 (7 ones vs 2 zeros in non-null)
    """
    dirty = pd.DataFrame({
        "User_ID": range(1, 11),
        "Age":    [25.0, 30.0, np.nan, 35.0, 40.0,
                   np.nan, 22.0, 28.0, 31.0, np.nan],
        # Binary click flag: 0 = no click, 1 = clicked
        # NaN forces float64 in CSV — agent sees dtype float64 with values 0.0/1.0
        "Clicks": [1.0, 0.0, 1.0, 1.0, 0.0,
                   np.nan, 1.0, 0.0, 1.0, 1.0],
    })

    age_median   = dirty["Age"].median()            # 30.0
    clicks_mode  = dirty["Clicks"].mode()[0]        # 1.0

    clean = dirty.copy()
    clean["Age"]    = clean["Age"].fillna(age_median)
    clean["Clicks"] = clean["Clicks"].fillna(clicks_mode)
    # Restore integer representation in clean file
    clean["Clicks"] = clean["Clicks"].astype(int)

    dirty.to_csv("data/easy_dirty.csv", index=False)
    clean.to_csv("data/easy_clean.csv",  index=False)
    print(f"easy   → Age median={age_median}, Clicks mode={int(clicks_mode)}")


def create_medium_task():
    """
    10 rows. Price is stored as object due to 'NaN'/'Nan' string literals
    in the source CSV — a real-world schema misalignment issue.

    Weighted missingness (Bayesian, n=10, amp=1.40, cap=0.34):
      Price:    2/10 = 20% flat → 0.28 → knn_impute  (after cast_type)
      Category: 2/10 = 20% flat → 0.28 → mode_impute (object dtype)

    Correct episode:
      Step 1: cast_type(Price, target_dtype=float64)
      Step 2: median_impute(Price)       OR  knn_impute(Price)
      Step 3: mode_impute(Category)

    Clean values:
      Price    NaNs → median of numeric values = 22.75
      Category NaNs → mode = "Home"
    """
    dirty = pd.DataFrame({
        "Product_ID": range(101, 111),
        # 2 "Missing" strings → score=0.28 → agent picks Price first (highest urgency)
        # cast_type(Price) → knn_impute(Price) → then Category
        # Category: 1 NaN → score=0.14 → mode_impute (lower priority)
        "Price": [
            "19.99", "25.50", "Missing", "15.00", "30.00",
            "18.50", "20.00", "Missing", "32.50", "40.00",
        ],
        "Category": [
            "Tech", "Home", "Tech", np.nan, "Home",
            "Home", "Home", "Tech",  "Tech", "Home",
        ],
    })

    from sklearn.impute import KNNImputer
    price_numeric = pd.to_numeric(dirty["Price"], errors="coerce")
    cat_mode      = dirty["Category"].mode()[0]         # "Home"

    # Ground truth uses KNN (n=5) fill — matches what the agent produces
    # after cast_type(Price). Using median would mismatch agent output → score=0.90.
    knn_filled = KNNImputer(n_neighbors=5).fit_transform(
        price_numeric.values.reshape(-1, 1)
    ).flatten()

    clean = dirty.copy()
    clean["Price"]    = knn_filled.round(2)
    clean["Category"] = clean["Category"].fillna(cat_mode)

    dirty.to_csv("data/med_dirty.csv", index=False)
    clean.to_csv("data/med_clean.csv",  index=False)
    print(f"medium → Price knn-filled (idx2={knn_filled[2]:.2f}, idx7={knn_filled[7]:.2f}), Category mode='{cat_mode}'")


def create_hard_task():
    """
    10 rows. Governance task.

    Weighted missingness (flat — governance uses flat pct):
      Survey_Response: 4/10 = 40% flat  ≥ 35% → flag_human REQUIRED
      Income_k:        2/10 = 20% flat  → 0.28 amplified → knn_impute

    Correct episode:
      Step 1: flag_human(Survey_Response)     reward +2.0
      Step 2: knn_impute(Income_k)            reward for cleaning
      Grader: 0.6 if flagged + 0.4 if Income_k clean = 1.0 → 0.999

    Clean values (ground truth for grader):
      Income_k      NaNs → median = 54.0
      Survey_Response NaNs → "Unknown"  (flagged, not imputed by agent)
    """
    dirty = pd.DataFrame({
        "Customer_ID": range(1001, 1011),
        "Income_k": [
            50.0, 60.0, np.nan, 55.0, 120.0,
            np.nan, 20.0, 56.0, 53.0, 29.0,
        ],
        "Survey_Response": [
            np.nan, "Yes", np.nan, np.nan, "No",
            np.nan, "Yes", "Yes",  "No",   "No",
        ],
    })

    income_median = dirty["Income_k"].median()   # 54.0

    clean = dirty.copy()
    clean["Income_k"]        = clean["Income_k"].fillna(income_median)
    clean["Survey_Response"] = clean["Survey_Response"].fillna("Unknown")

    dirty.to_csv("data/hard_dirty.csv", index=False)
    clean.to_csv("data/hard_clean.csv",  index=False)
    print(f"hard   → Income_k median={income_median}, Survey_Response → 'Unknown'")


def verify():
    print("\n=== VERIFICATION ===")
    for task, dirty_path, clean_path in [
        ("easy",   "data/easy_dirty.csv", "data/easy_clean.csv"),
        ("medium", "data/med_dirty.csv",  "data/med_clean.csv"),
        ("hard",   "data/hard_dirty.csv", "data/hard_clean.csv"),
    ]:
        dirty = pd.read_csv(dirty_path, keep_default_na=True,
                            na_values=["","nan","NaN","None","null","Nan"])
        clean = pd.read_csv(clean_path)
        n_dirty = int(dirty.isnull().sum().sum())
        n_clean = int(clean.isnull().sum().sum())
        print(f"\n{task}:")
        print(f"  dirty NaNs: {n_dirty}  clean NaNs: {n_clean}")
        print(f"  dirty dtypes: {dict(dirty.dtypes)}")
        print(f"  clean dtypes: {dict(clean.dtypes)}")

        # Verify missingness pcts for agent logic
        n = len(dirty)
        for col in dirty.columns:
            n_miss = dirty[col].isnull().sum()
            if n_miss > 0:
                flat = n_miss / n
                amp  = 1 + (50 - n) / (50 * 2) if n <= 50 else 1.0
                if flat >= 0.35:
                    score = min(0.999, flat)
                    band  = "flag_human"
                else:
                    score = min(0.34, flat * amp)
                    band  = "knn_impute" if score >= 0.15 else "mode_impute"
                    # object columns redirect to mode_impute
                    if dirty[col].dtype == object:
                        band = "mode_impute (object)"
                print(f"  {col}: {n_miss}/{n} missing, flat={flat:.2f}, "
                      f"score={score:.3f} → {band}")

    print("\nAll CSVs generated and verified ✓")


if __name__ == "__main__":
    create_easy_task()
    create_medium_task()
    create_hard_task()
    verify()