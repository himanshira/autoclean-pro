"""Core cleaning logic and reward calculations for AutoClean-Pro."""

import pandas as pd
import numpy as np
from typing import Tuple, List, Any

def calculate_cleaning_gain(old_df: pd.DataFrame, new_df: pd.DataFrame) -> float:
    """Calculates improvement in data quality.
    Args:
        old_df: The DataFrame before the action.
        new_df: The DataFrame after the action.
    Returns:
        A float representing the quality gain (0.0 to 1.0).
    """
    old_nulls = old_df.isnull().sum().sum()
    new_nulls = new_df.isnull().sum().sum()

    # Simple gain: normalized reduction in nulls
    if old_nulls == 0:
        return 0.0
    return max(0.0, (old_nulls - new_nulls) / old_nulls)

def calculate_rarity_bonus(history: List[str], current_tool: str) -> float:
    """Award a bonus for non-repetitive, expert-level sequences.
    Args:
        history: List of tools used in the current episode.
        current_tool: The tool currently being used.
    Returns:
        A bonus value (e.g., 0.05).  
    """
    if not history:
        return 0.0
    
    # Penalty for repetition (Avoids infinite loops)
    if current_tool == history[-1]:
        return -0.1

    # Bonus for "Self-Correction" (using a different tool after a failure)
    if len(history) > 1 and current_tool not in history[-3:]:
        return 0.05
    return 0.0
"""Constraint validation and imputation logic for AutoClean-Pro."""

import pandas as pd
from typing import Tuple, Optional

def validate_cleaning_strategy(df: pd.DataFrame, column: str) -> Tuple[str, Optional[str]]:
    """Validates the strategy based on missing value percentage.
    
    Args:
        df: The current DataFrame.
        column: The target column.
        
    Returns:
        A tuple of (strategy_label, recommendation).
    """
    missing_pct = df[column].isnull().mean()
    
    if missing_pct >= 0.50:
        return "HUMAN_REDIRECT", "Too much missing data for autonomous cleaning."
    
    if 0.05 <= missing_pct <= 0.25:
        # Suggest specific imputation based on data type
        if pd.api.types.is_numeric_dtype(df[column]):
            return "AUTO_IMPUTE", "Recommend Median or KNN Imputation."
        else:
            return "AUTO_IMPUTE", "Recommend Mode Imputation."
            
    return "STANDARD", "Proceed with basic cleaning."