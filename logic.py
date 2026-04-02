import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

def get_weighted_missing_report(df: pd.DataFrame, weights: np.ndarray) -> Dict[str, float]:
    report = {}
    total_weight = weights.sum() 
    for col in df.columns:
        is_null = df[col].isna()
        weighted_null_sum = weights[is_null].sum()
        report[col] = float(weighted_null_sum / total_weight) if total_weight > 0 else 0.0
    return report

def calculate_cleaning_gain(old_df: pd.DataFrame, new_df: pd.DataFrame, column: str, tool: str, weights: np.ndarray, target_df: pd.DataFrame = None) -> float:
    if column not in old_df.columns:
        return -0.5
    
    is_null_before = old_df[column].isnull()
    total_weight = weights.sum()
    weighted_missing_pct = weights[is_null_before].sum() / total_weight if total_weight > 0 else 0.0
    
    col_lower = column.lower()
    is_id_col = any(k in col_lower for k in ['id', 'idx', 'key'])

    # 1. GOVERNANCE (Aligned with Hard Task)
    if weighted_missing_pct >= 0.35:
        return 2.0 if tool == "flag_human" else -5.0
    
    # 2. ID HANDLING (Prioritize casting at start)
    if is_id_col:
        if tool == "cast_type" and new_df[column].dtype == object and old_df[column].dtype != object:
            return 1.5 # High reward to get this out of the way first
        return 0.1 

    # 3. PROGRESS CALCULATION
    old_nulls = old_df[column].isnull().sum()
    new_nulls = new_df[column].isnull().sum()
    if old_nulls == 0: return 0.0 

    gain = float((old_nulls - new_nulls) / old_nulls)
    
    # 4. STRATEGY ALIGNMENT BONUS (Mapped to your Imputation Rules)
    strategy_bonus = 0.0
    is_numeric = np.issubdtype(old_df[column].dtype, np.number)

    if not is_numeric or old_df[column].dtype == 'object':
        # Categorical Logic
        if tool == "mode_impute" and weighted_missing_pct < 0.35:
            strategy_bonus = 0.75 # Boosted to ensure it beats generic median
    else:
        # Numeric Logic
        if 0.05 <= weighted_missing_pct < 0.15 and tool == "median_impute":
            strategy_bonus = 0.5
        elif 0.15 <= weighted_missing_pct < 0.35 and tool == "knn_impute":
            strategy_bonus = 0.5
        elif weighted_missing_pct < 0.05 and tool == "median_impute":
            strategy_bonus = 0.3

    # 5. ACCURACY BONUS (Float Jitter Protection)
    accuracy_bonus = 0.0
    if target_df is not None:
        newly_filled = old_df[column].isnull() & new_df[column].notnull()
        if newly_filled.any():
            # Use a small epsilon for float comparison or round both to 2
            val_new = new_df.loc[newly_filled, column].astype(float).round(2) if is_numeric else new_df.loc[newly_filled, column].astype(str)
            val_tar = target_df.loc[newly_filled, column].astype(float).round(2) if is_numeric else target_df.loc[newly_filled, column].astype(str)
            
            if (val_new == val_tar).all():
                accuracy_bonus = 1.0

    return gain + strategy_bonus + accuracy_bonus

def calculate_rarity_bonus(history: List[str], current_tool: str) -> float:
    if len(history) < 2: return 0.0
    # Penalty for back-to-back same tool usage (Stuck behavior)
    if current_tool == history[-2]:
        return -2.0 
    return 0.1

def validate_cleaning_strategy(df: pd.DataFrame, column: str, weights: np.ndarray) -> Tuple[str, str]:
    # SYNCED WITH GOVERNANCE
    is_null = df[column].isnull()
    weighted_pct = weights[is_null].sum() / weights.sum() if weights.sum() > 0 else 0.0
    
    if weighted_pct >= 0.35:
        return "FLAG", f"Missingness {weighted_pct:.1%}. Governance: flag_human."
    if weighted_pct >= 0.15:
        return "KNN", "Use knn_impute."
    return "MEDIAN/MODE", "Use median (num) or mode (cat)."