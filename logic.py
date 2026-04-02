import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

def get_weighted_missing_report(df: pd.DataFrame, weights: np.ndarray) -> Dict[str, float]:
    """Calculates missingness where each row's contribution is its Bayesian weight."""
    report = {}
    total_weight = weights.sum() 
    for col in df.columns:
        is_null = df[col].isna()
        weighted_null_sum = weights[is_null].sum()
        # Correct Bayesian Normalization
        report[col] = float(weighted_null_sum / total_weight) if total_weight > 0 else 0.0
    return report

def calculate_cleaning_gain(old_df: pd.DataFrame, new_df: pd.DataFrame, column: str, tool: str, weights: np.ndarray, target_df: pd.DataFrame = None) -> float:
    """Calculates reward based on progress, threshold, and ID governance."""
    # SAfety check for halucinated columns 
    if column not in old_df.columns:
        return -0.5
    is_null_before = old_df[column].isnull()
    total_weight = weights.sum()
    weighted_missing_pct = weights[is_null_before].sum() / total_weight if total_weight > 0 else 0.0
    
    col_lower = column.lower()
    id_keywords = ['id', 'idx', 'key']
    is_id_col = any(keyword in col_lower for keyword in id_keywords)

    # 1. GOVERNANCE (Lowered to 40% to catch the Hard Task Survey_Response)
    # This ensures the Agent flags the column instead of trying to impute 40% missing data.
    # In logic.py
    if weighted_missing_pct >= 0.35: # Changed from 0.40 to 0.35
        return 2.0 if tool == "flag_human" else -5.0
    
    # 2. ID HANDLING
    if is_id_col:
        # If the agent casts it, give a reward. If it imputes, we stay neutral 
        # because the grader ignores IDs anyway.
        if tool == "cast_type" and new_df[column].dtype == object and old_df[column].dtype != object:
            return 1.0
        return 0.1 

    # 3. PROGRESS CALCULATION (For Data Columns)
    old_nulls = old_df[column].isnull().sum()
    new_nulls = new_df[column].isnull().sum()
    
    if old_nulls == 0:
        return 0.0 

    # Base gain: what % of NaNs were removed?
    gain = float((old_nulls - new_nulls) / old_nulls)
    
    # 4. STRATEGY ALIGNMENT BONUS
    # Reward using the specific tools defined in your roadmap
    strategy_bonus = 0.0
    if 0.05 <= weighted_missing_pct < 0.15 and tool == "median_impute":
        strategy_bonus = 0.5
    elif 0.15 <= weighted_missing_pct < 0.40 and tool == "knn_impute":
        strategy_bonus = 0.5

    # 5. ACCURACY BONUS (If target_df is provided)
    # If the new values perfectly match the ground truth, give an extra 1.0
    accuracy_bonus = 0.0
    if target_df is not None:
        # Compare only the rows that were originally null
        newly_filled = old_df[column].isnull() & new_df[column].notnull()
        if newly_filled.any():
            # Check if all newly filled values match the target (as strings to avoid float jitter)
            matches = (new_df.loc[newly_filled, column].astype(str) == 
                       target_df.loc[newly_filled, column].astype(str)).all()
            if matches:
                accuracy_bonus = 1.0

    return gain + strategy_bonus + accuracy_bonus

def calculate_rarity_bonus(history: List[str], current_tool: str) -> float:
    """Rewards variety and penalizes loops/stuck behavior."""
    if len(history) < 2: return 0.0
    
    # history[-1] is the current tool because we append BEFORE calling this
    # So we compare against history[-2]
    if current_tool == history[-2]:
        return -1.5 
    
    # Variety bonus
    return 0.1

def validate_cleaning_strategy(df: pd.DataFrame, column: str, weights: np.ndarray) -> Tuple[str, str]:
    """Helper used for debugging and verifying Agent decisions."""
    is_null = df[column].isnull()
    total_weight = weights.sum()
    weighted_pct = weights[is_null].sum() / total_weight if total_weight > 0 else 0.0
    col_lower = str(column).lower()
    
    if weighted_pct >= 0.50:
        return "FLAG", "Weighted missingness >= 50%. Governance requires flag_human."

    if any(idx in col_lower for idx in ['id', 'idx', 'key']):
        return "SKIP", "Identifier. Grader ignores this; move to data columns."

    if weighted_pct >= 0.15:
        return "KNN", f"Weighted missingness {weighted_pct:.1%}. Use knn_impute."
    
    return "MEDIAN", "Weighted missingness < 15%. Use median_impute."