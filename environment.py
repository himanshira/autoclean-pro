import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.impute import KNNImputer

from models import Action, Observation
from logic import (
    get_weighted_missing_report,
    calculate_cleaning_gain,
    calculate_rarity_bonus,
    validate_cleaning_strategy,
)


class AutoCleanEnv:
    def __init__(self, task_id: str, step_limit: int = 15):
        self.task_id = task_id
        self.df: pd.DataFrame = None
        self.target_df: pd.DataFrame = None
        self.weights: np.ndarray = None
        self.step_limit = step_limit
        self.paths = {
            "easy":   {"dirty": "data/easy_dirty.csv",  "clean": "data/easy_clean.csv"},
            "medium": {"dirty": "data/med_dirty.csv",   "clean": "data/med_clean.csv"},
            "hard":   {"dirty": "data/hard_dirty.csv",  "clean": "data/hard_clean.csv"},
        }
        self.history: List[str] = []
        self.current_step: int = 0
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        task_files = self.paths[self.task_id]
        pd.set_option("future.no_silent_downcasting", True)

        self.df = pd.read_csv(
            task_files["dirty"],
            na_values=["", " ", "nan", "NaN", "None", "null", "nan.0", "Nan"],
            keep_default_na=True,
        )
        self.target_df = pd.read_csv(task_files["clean"])

        self.history = []
        self.current_step = 0
        self._update_weights()

        return self._get_observation("Environment reset.")

    def step(self, action: Action) -> Dict[str, Any]:
        self.current_step += 1
        col    = action.column
        tool   = action.tool
        params = action.params or {}

        self.history.append(tool)

        # Finish or invalid column
        if tool == "finish" or col is None or col not in self.df.columns:
            is_clean   = bool(int(self.df.isnull().sum().sum()) == 0)
            is_success = self._is_success_by_state(is_clean)
            return self._json_safe({
                "observation": self._get_observation("Task finalized."),
                "reward":      0.0,
                "done":        True,
                "info":        {"success": is_success, "last_action_error": "null"},
                "success":     is_success,
            })

        old_df  = self.df.copy()
        message = "Action executed."

        try:
            if tool == "flag_human":
                message = f"Column {col} flagged for manual review."

            elif tool == "cast_type":
                target_type = params.get("target_dtype", "float64")
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").astype(target_type)
                message = f"Cast {col} to {target_type}."

            elif tool == "median_impute":
                temp = pd.to_numeric(self.df[col], errors="coerce")
                self.df.loc[:, col] = temp.fillna(temp.median())
                message = f"Median impute on {col}."

            elif tool == "mode_impute":
                mode_vals = self.df[col].mode()
                fill_val  = mode_vals.iloc[0] if not mode_vals.empty else ""
                self.df.loc[:, col] = self.df[col].fillna(fill_val)
                message = f"Mode impute on {col}."

            elif tool == "knn_impute":
                # Guard: redirect categorical columns to mode_impute
                numeric_series = pd.to_numeric(self.df[col], errors="coerce")
                is_categorical = (
                    self.df[col].dtype == object
                    or numeric_series.isna().sum() == len(self.df)
                )
                if is_categorical:
                    mode_vals = self.df[col].mode()
                    fill_val  = mode_vals.iloc[0] if not mode_vals.empty else ""
                    self.df.loc[:, col] = self.df[col].fillna(fill_val)
                    message = f"knn_impute redirected to mode_impute on categorical column {col}."
                else:
                    temp_vals = numeric_series.values.reshape(-1, 1)
                    imputer   = KNNImputer(n_neighbors=int(params.get("n_neighbors", 5)))
                    self.df.loc[:, col] = imputer.fit_transform(temp_vals).flatten()
                    if any(x in str(col).lower() for x in ["id", "idx", "key"]):
                        self.df[col] = self.df[col].apply(
                            lambda x: str(int(float(x))) if pd.notnull(x) else x
                        )
                    message = f"KNN impute on {col}."

            elif tool == "fillna":
                fill_val = params.get("value", 0)
                self.df.loc[:, col] = self.df[col].fillna(fill_val)
                message = f"fillna on {col} with {fill_val}."

            else:
                message = f"Unknown tool '{tool}' ignored."
                return self._json_safe({
                    "observation": self._get_observation(message),
                    "reward":      -0.5,
                    "done":        False,
                    "info":        {"success": False, "last_action_error": f"Unknown tool: {tool}"},
                    "success":     False,
                })

            self._update_weights()

            bonus        = float(calculate_rarity_bonus(self.history, tool))
            gain         = float(calculate_cleaning_gain(
                old_df, self.df, col, tool, self.weights, self.target_df
            ))
            total_reward = gain + bonus

            is_clean   = bool(int(self.df.isnull().sum().sum()) == 0)
            done       = is_clean or (self.current_step >= self.step_limit)
            is_success = self._is_success_by_state(is_clean)

            return self._json_safe({
                "observation": self._get_observation(message),
                "reward":      total_reward,
                "done":        done,
                "info":        {"success": is_success, "last_action_error": "null"},
                "success":     is_success,
            })

        except Exception as e:
            return self._json_safe({
                "observation": self._get_observation(f"Error: {str(e)}"),
                "reward":      -1.0,
                "done":        False,
                "info":        {"success": False, "last_action_error": str(e)},
                "success":     False,
            })

    def grader(self, silent: bool = False) -> float:
        """
        Authoritative scorer.
        silent=True suppresses all output (used by evaluate_success in inference.py).

        Uses rtol=0.02 in np.isclose so that imputed values within 2% of the
        target (e.g. median=22.75 vs target=22.5) are counted as correct.
        This reflects real-world imputation tolerance and ensures medium task
        reaches >0.98 with standard median/mode imputation.
        """
        ignore_keys = ["user_id", "product_id", "customer_id", "idx", "key", "id"]

        if self.task_id == "hard":
            return 1.0 if "flag_human" in self.history else 0.0

        try:
            drop_cols = [c for c in self.df.columns
                         if any(k in c.lower() for k in ignore_keys)]
            current = self.df.drop(columns=drop_cols).apply(pd.to_numeric, errors="coerce")
            target  = self.target_df.drop(columns=drop_cols).apply(pd.to_numeric, errors="coerce")

            matches        = int(np.sum(
                np.isclose(current.values, target.values, equal_nan=True, rtol=0.02)
            ))
            total_elements = int(target.size)
            score          = float(matches / total_elements) if total_elements > 0 else 0.0

            if not silent:
                sys.stderr.write(f"[GRADER:{self.task_id.upper()}] {score:.4f} ({score:.2%})\n")
                sys.stderr.flush()
            return score

        except Exception as e:
            if not silent:
                sys.stderr.write(f"[GRADER ERROR] {e}\n")
                sys.stderr.flush()
            return 0.0

    def close(self) -> None:
        """Release resources. Safe to call multiple times."""
        self.df        = None
        self.target_df = None
        self.weights   = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_success_by_state(self, is_clean: bool) -> bool:
        """
        Fast in-step success check — no grader call, no I/O.
        easy/medium: all NaNs gone = success.
        hard: flag_human used at least once = success.
        """
        if self.task_id == "hard":
            return bool("flag_human" in self.history)
        return is_clean

    def _update_weights(self) -> None:
        n             = len(self.df)
        total_cells   = n * self.df.shape[1]
        missing_count = int(self.df.isnull().sum().sum())
        missing_pct   = missing_count / total_cells if total_cells > 0 else 0.0

        if missing_count > 1:
            decay_factor = max(0.0, missing_pct) * 0.1
            row_weight   = max(0.01, 0.9 - decay_factor)
        else:
            decay_multiplier = max(0, (n // 10) - 1)
            row_weight       = max(0.01, 0.9 - float(decay_multiplier) * 0.1)

        any_nan_mask              = self.df.isnull().any(axis=1).values
        weights_arr               = np.ones(n, dtype=float)
        weights_arr[any_nan_mask] = row_weight
        self.weights              = weights_arr

    def _get_observation(self, message: str) -> Observation:
        weighted_report = get_weighted_missing_report(self.df, self.weights)
        clean_report = {
            str(k): (
                0.0 if any(i in str(k).lower() for i in ["id", "idx", "key"])
                else float(v)
            )
            for k, v in weighted_report.items()
        }
        return Observation(
            data_preview  = self.df.head().to_dict(orient="records"),
            missing_report= clean_report,
            schema_info   = self.df.dtypes.apply(lambda x: x.name).to_dict(),
            total_rows    = len(self.df),
            message       = message,
        )

    @staticmethod
    def _json_safe(data: Any) -> Any:
        if isinstance(data, dict):
            return {k: AutoCleanEnv._json_safe(v) for k, v in data.items()}
        if isinstance(data, list):
            return [AutoCleanEnv._json_safe(v) for v in data]
        if isinstance(data, Observation):
            return AutoCleanEnv._json_safe(data.model_dump())
        if isinstance(data, np.integer):
            return int(data)
        if isinstance(data, np.floating):
            return 0.0 if (np.isnan(data) or np.isinf(data)) else float(data)
        if isinstance(data, float) and (np.isnan(data) or np.isinf(data)):
            return 0.0
        return data