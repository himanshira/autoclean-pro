import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.impute import KNNImputer

from models import Action, Observation  # Ensure Reward/State are used or removed

class AutoCleanEnv:
    def __init__(self, task_id: str, step_limit: int = 15):
        self.task_id = task_id
        self.step_limit = step_limit
        self.paths = {
            "easy": {"dirty": "data/easy_dirty.csv", "clean": "data/easy_clean.csv"},
            "medium": {"dirty": "data/med_dirty.csv", "clean": "data/med_clean.csv"},
            "hard": {"dirty": "data/hard_dirty.csv", "clean": "data/hard_clean.csv"}
        }
        self.reset()

    def reset(self) -> Observation:
        """Resets the environment and PREVENTS state leakage."""
        task_files = self.paths[self.task_id]
        # Fixed typo: keep_default_na
        self.df = pd.read_csv(task_files["dirty"],
                              na_values=['', ' ', 'nan', 'NaN', 'None', 'null'],
                              keep_default_na=True)
        self.target_df = pd.read_csv(task_files["clean"])

        self.history = []
        self.current_step = 0
        return self._get_observation("Environment reset. Task started.")

    def step(self, action: Action) -> tuple:
        # 0. INITIALIZE TRACKING VARIABLES
        self.current_step += 1
        old_df = self.df.copy() # Fixed: Added old_df for reward calculation
        info = {}               # Fixed: Initialized info
        total_reward = 0.0      # Fixed: Initialized total_reward
        done = False
        message = ""

        # 1. VALIDATE COLUMN (Essential to prevent crashes)
        if action.tool != "finish" and (not action.column or action.column not in self.df.columns):
            msg = f"Invalid column: '{action.column}'"
            return self._get_observation(msg), -0.1, False, {"error": "bad_col"}

        # 2. GOVERNANCE CHECK (The 50% Rule)
        missing_pct = 0
        if action.column in self.df.columns:
            missing_pct = self.df[action.column].isnull().mean()

        if missing_pct >= 0.50 and action.tool not in ["flag_human", "finish"]:
            msg = f"GOVERNANCE BLOCKED: {action.column} has {missing_pct:.0%} missing data. Use flag_human."
            return self._get_observation(msg), -0.5, False, {"error": "HITL_REQUIRED"}

        # 3. TOOL EXECUTION
        try:
            if action.tool == "flag_human":
                self.history.append("flag_human")
                if missing_pct >= 0.50:
                    message = f"SUCCESS: {action.column} flagged for review."
                    total_reward = 1.0
                    if self.task_id == "hard":
                        done = True
                else:
                    message = f"FAILURE: {action.column} only has {missing_pct:.1%}. Flagging unnecessary."
                    total_reward = -0.2

            elif action.tool == "knn_impute":
                k = action.params.get("k", 5)
                message = self._apply_knn(action.column, k)
                self.history.append("knn_impute")

            elif action.tool == "mode_impute":
                message = self._apply_mode(action.column)
                self.history.append("mode_impute")

            elif action.tool == "fillna":
                val = action.params.get("value", 0)
                self.df[action.column] = self.df[action.column].fillna(val)
                message = f"Applied fillna to {action.column}."
                self.history.append("fillna")

            elif action.tool == "finish":
                done = True
                message = "Task submitted by agent."
                self.history.append("finish")

            # 4. REWARD CALCULATION (Only if not already set by flag_human)
            if action.tool not in ["flag_human", "finish"]:
                # Logic imports assumed from your script
                from logic import calculate_cleaning_gain, calculate_rarity_bonus
                gain = calculate_cleaning_gain(old_df, self.df)
                bonus = calculate_rarity_bonus(self.history, action.tool)
                total_reward = max(-1.0, min(1.0, gain + bonus))

        except Exception as e:
            message = f"RUNTIME ERROR: {str(e)}"
            return self._get_observation(message), -0.1, False, {"error": "EXCEPTION"}

        # 5. TERMINATION CHECK
        if self.df.equals(self.target_df) or self.current_step >= self.step_limit:
            done = True
            message += " [Episode Terminated]"

        return self._get_observation(message), total_reward, done, info

    def grader(self) -> float:
        """Deterministic Grader: Rewards Policy in Hard tasks & Math in Easy/Medium."""
        try:
            # 1. Hard Task: Governance-First Scoring
            if self.task_id == "hard":
                if "flag_human" in self.history:
                    return 1.0 
                return 0.0

            # 2. Easy/Medium: Mathematical Fidelity Scoring
            current = self.df.reset_index(drop=True)
            target = self.target_df.reset_index(drop=True)

            numeric_cols = current.select_dtypes(include=[np.number]).columns
            object_cols = current.select_dtypes(exclude=[np.number]).columns

            num_match = 0
            if not numeric_cols.empty:
                num_match = np.isclose(
                    current[numeric_cols].values, 
                    target[numeric_cols].values, 
                    equal_nan=True, atol=1e-4
                ).sum()

            obj_match = 0
            if not object_cols.empty:
                obj_match = (current[object_cols] == target[object_cols]).values.sum()

            return float((num_match + obj_match) / current.size)

        except Exception as e:
            print(f"Grader Error: {e}")
            return 0.0

    def _apply_knn(self, column: str, k: int) -> str:
        if not pd.api.types.is_numeric_dtype(self.df[column]):
             self.df[column] = pd.to_numeric(self.df[column], errors='coerce')
        
        imputer = KNNImputer(n_neighbors=k)
        self.df[[column]] = imputer.fit_transform(self.df[[column]])
        return f"Applied KNN to {column}."

    def _apply_mode(self, column: str) -> str:
        if self.df[column].isnull().all():
            return "Failed: Column is entirely empty."
        mode_val = self.df[column].mode()[0]
        self.df[column] = self.df[column].fillna(mode_val)
        return f"Applied Mode ({mode_val}) to {column}."

    def _get_observation(self, message: str) -> Observation:
        missing_counts = self.df.isnull().sum().to_dict()
        preview_list = self.df.head().to_dict(orient='records')
        schema_dict = self.df.dtypes.apply(lambda x: x.name).to_dict()

        return Observation(
            data_preview=preview_list,
            missing_report=missing_counts,
            schema_info=schema_dict,
            message=message
        )