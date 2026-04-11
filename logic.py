import pandas as pd
import numpy as np
from typing import Tuple, List, Dict


# ---------------------------------------------------------------------------
# Per-column Bayesian-amplified missingness report
# ---------------------------------------------------------------------------

def get_weighted_missing_report(
    df: pd.DataFrame,
    weights: np.ndarray,           # kept for API compatibility — not used internally
    bayesian_mode: str = "auto",   # "auto" | "on" | "off"
    scarce_threshold: int = 50,
) -> Dict[str, float]:
    """
    Compute per-column weighted missingness scores.

    Modes
    -----
    "off"  — flat proportion: n_missing / n_rows  (standard datasets)
    "on"   — Bayesian amplification always applied
    "auto" — amplification applied only when n_rows <= scarce_threshold

    Amplification formula (scarce mode)
    ------------------------------------
    amp = 1 + (scarce_threshold - n) / (scarce_threshold * 2)
      n=10, threshold=50 → amp = 1.40
      n=30, threshold=50 → amp = 1.20
      n=50, threshold=50 → amp = 1.00  (boundary: no boost)

    Tool-selection bands preserved
    --------------------------------
    1 missing / 10 rows → flat 0.10 × 1.40 = 0.14 → median_impute
    2 missing / 10 rows → flat 0.20 × 1.40 = 0.28 → knn_impute
    3 missing / 10 rows → flat 0.30 × 1.40 = 0.42 → flag_human
    """
    n = len(df)
    ignore = {"id", "idx", "key"}

    # Determine effective mode
    if bayesian_mode == "auto":
        use_bayesian = (n <= scarce_threshold)
    elif bayesian_mode == "on":
        use_bayesian = True
    else:
        use_bayesian = False

    amp = 1.0
    if use_bayesian:
        amp = 1.0 + (scarce_threshold - n) / (scarce_threshold * 2)
        amp = max(1.0, amp)   # never reduce

    # Sentinel strings that count as effectively missing in object columns
    _SENTINELS = {"missing", "n/a", "na", "none", "null", "unknown", "?", "-", ""}

    def _count_missing(series: pd.Series) -> int:
        """Count nulls + sentinel strings in object columns."""
        null_count = int(series.isnull().sum())
        if str(series.dtype) not in ("object", "str", "string"):
            return null_count
        extra = int(series.dropna().apply(
            lambda x: str(x).strip().lower() in _SENTINELS
        ).sum())
        return null_count + extra

    report: Dict[str, float] = {}
    for col in df.columns:
        # Zero-out ID columns so agent never wastes steps on them
        if any(k in col.lower() for k in ignore):
            report[col] = 0.0
            continue

        n_missing = _count_missing(df[col])
        if n_missing == 0:
            report[col] = 0.0
            continue

        flat_pct = n_missing / n

        # Two-regime scoring:
        # 1. Governance regime (flat_pct >= 0.35): return flat_pct directly.
        #    Amplification must NEVER push a sub-threshold column over 0.35.
        #    Governance is about actual data loss, not perceived importance.
        # 2. Tool-selection regime (flat_pct < 0.35): amplify, but cap at 0.34
        #    so the score never crosses the governance boundary.
        #    This shifts median→knn when appropriate in scarce datasets,
        #    without ever triggering a false flag_human.
        if flat_pct >= 0.35:
            report[col] = float(min(0.999, flat_pct))
        else:
            amplified = flat_pct * amp
            # Hard cap: amplification cannot push a column into governance territory
            report[col] = float(min(0.34, amplified))

    return report


# ---------------------------------------------------------------------------
# Cleaning gain — column-level reward signal
# ---------------------------------------------------------------------------

def calculate_cleaning_gain(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    column: str,
    tool: str,
    weights: np.ndarray,           # kept for API compatibility
    target_df: pd.DataFrame = None,
) -> float:
    if column not in old_df.columns:
        return -0.5

    is_null_before = old_df[column].isnull()
    n = len(old_df)
    weighted_missing_pct = is_null_before.sum() / n if n > 0 else 0.0

    col_lower = column.lower()
    is_id_col = any(k in col_lower for k in ["id", "idx", "key"])

    # 1. Governance gate
    if weighted_missing_pct >= 0.35:
        return 2.0 if tool == "flag_human" else -5.0

    # 2. ID columns
    if is_id_col:
        return 0.1

    # 3. Progress
    old_nulls = int(old_df[column].isnull().sum())
    new_nulls = int(new_df[column].isnull().sum())
    if old_nulls == 0:
        return 0.0

    gain = float((old_nulls - new_nulls) / old_nulls)

    # 4. Strategy alignment bonus
    strategy_bonus = 0.0
    is_numeric = np.issubdtype(old_df[column].dtype, np.number)

    if not is_numeric or old_df[column].dtype == object:
        if tool == "mode_impute" and weighted_missing_pct < 0.35:
            strategy_bonus = 0.75
    else:
        if 0.05 <= weighted_missing_pct < 0.15 and tool == "median_impute":
            strategy_bonus = 0.5
        elif 0.15 <= weighted_missing_pct < 0.35 and tool == "knn_impute":
            strategy_bonus = 0.5
        elif weighted_missing_pct < 0.05 and tool == "median_impute":
            strategy_bonus = 0.3

    # 5. Accuracy bonus
    accuracy_bonus = 0.0
    if target_df is not None:
        newly_filled = old_df[column].isnull() & new_df[column].notnull()
        if newly_filled.any():
            try:
                if is_numeric:
                    val_new = new_df.loc[newly_filled, column].astype(float).round(2)
                    val_tar = target_df.loc[newly_filled, column].astype(float).round(2)
                else:
                    val_new = new_df.loc[newly_filled, column].astype(str)
                    val_tar = target_df.loc[newly_filled, column].astype(str)
                if (val_new == val_tar).all():
                    accuracy_bonus = 1.0
            except Exception:
                pass

    return gain + strategy_bonus + accuracy_bonus


# ---------------------------------------------------------------------------
# Rarity / repetition bonus
# ---------------------------------------------------------------------------

def calculate_rarity_bonus(history: List[str], current_tool: str) -> float:
    if len(history) < 2:
        return 0.0
    if current_tool == history[-2]:
        return -2.0   # penalise back-to-back same tool
    return 0.1


# ---------------------------------------------------------------------------
# Strategy advisor — used by inference.py to build LLM hint
# ---------------------------------------------------------------------------

def validate_cleaning_strategy(
    df: pd.DataFrame,
    column: str,
    weights: np.ndarray,           # kept for API compatibility
    bayesian_mode: str = "auto",
    scarce_threshold: int = 50,
) -> Tuple[str, str]:
    """
    Return (strategy_name, reason) for the given column
    using the same amplification logic as get_weighted_missing_report.
    """
    report = get_weighted_missing_report(
        df, weights,
        bayesian_mode=bayesian_mode,
        scarce_threshold=scarce_threshold,
    )
    weighted_pct = report.get(column, 0.0)

    if weighted_pct >= 0.35:
        return "FLAG", f"Weighted missingness {weighted_pct:.1%} ≥ 35% → flag_human required."
    if weighted_pct >= 0.15:
        return "KNN", f"Weighted missingness {weighted_pct:.1%} in 15-34% band → knn_impute."
    if weighted_pct > 0:
        return "MEDIAN_OR_MODE", f"Weighted missingness {weighted_pct:.1%} < 15% → median/mode impute."
    return "CLEAN", "No missing values in this column."