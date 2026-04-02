import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.impute import KNNImputer
from logic import get_weighted_missing_report, calculate_cleaning_gain, calculate_rarity_bonus
from models import Action, Observation

class AutoCleanEnv:
    def __init__(self, task_id: str, step_limit: int = 15):
        self.task_id = task_id
        self.df = None
        self.target_df = None
        self.weights = None 
        self.step_limit = step_limit
        self.paths = {
            "easy": {"dirty": "data/easy_dirty.csv", "clean": "data/easy_clean.csv"},
            "medium": {"dirty": "data/med_dirty.csv", "clean": "data/med_clean.csv"},
            "hard": {"dirty": "data/hard_dirty.csv", "clean": "data/hard_clean.csv"}
        }
        self.history = []
        self.current_step = 0
        self.reset()

    def reset(self) -> Observation:
        task_files = self.paths[self.task_id]
        # Silencing the downcasting warning globally for clean logs
        pd.set_option('future.no_silent_downcasting', True)
        # Load data with standard NA handling
        self.df = pd.read_csv(
            task_files["dirty"],
            na_values=['', ' ', 'nan', 'NaN', 'None', 'null', 'nan.0'],
            keep_default_na=True
        )
        self.target_df = pd.read_csv(task_files["clean"])
        
        self.history = []
        self.current_step = 0
        
        # Initialize Bayesian weights immediately upon load
        self._update_weights()
        
        return self._get_observation("Environment reset.")

    def _update_weights(self):
        """Internal logic to refresh Bayesian weights based on current state."""
        n = len(self.df)
        decay_multiplier = (n // 10) - 1

        # Calculate cell-level missingness density
        total_cells = n * self.df.shape[1]
        missing_count = self.df.isnull().sum().sum()
        missing_pct = missing_count / total_cells if total_cells > 0 else 0

        # Calculate the row_weight for NaN rows
        if missing_count > 1:
            decay_factor = max(0, missing_pct) * 0.1
            row_weight = max(0.01, 0.9 - decay_factor)
        else:
            decay_factor = max(0, decay_multiplier) * 0.1
            row_weight = max(0.01, 0.9 - decay_factor)

        # Apply weights: 1.0 for valid data, row_weight for rows with ANY NaN
        self.weights = pd.Series(1.0, index=self.df.index)
        any_nan_mask = self.df.isnull().any(axis=1)
        self.weights[any_nan_mask] = row_weight
        
        return row_weight
    
    def step(self, action: Action) -> tuple:
        self.current_step += 1
        col = action.column
        tool = action.tool
        params = action.params 
        
        # 1. Immediate History Logging (Crucial for Grader/Governance)
        self.history.append(tool)

        # 2. Handle Termination or Invalid Columns
        # We cast the 'done' boolean to a standard Python bool to avoid FastAPI serialization errors
        if tool == "finish" or col == "None" or col not in self.df.columns:
            is_finished = bool(tool == "finish")
            return self._get_observation("Step skipped or Task completed."), 0.0, is_finished, {}

        # 3. Snapshot current state for Reward Calculation
        old_df = self.df.copy()

        try:
            # 4. Execute Tool Logic (flag_human at top for priority)
            if tool == "flag_human":
                self.df[col] = self.df[col].astype(object)
                # ALIGNED WITH generate_data.py Ground Truth
                self.df[col] = self.df[col].fillna("Review Required")
                message = f"Flagged {col} for manual review."

            elif tool == "cast_type":
                target_type = params.get("type", "object")
                if target_type == "object":
                    # Clean conversion: preserves NaNs while ensuring object dtype
                    self.df[col] = self.df[col].astype(str).replace(['nan', 'None', 'NaN', 'np.nan'], np.nan)
                else:
                    # Safer numeric casting for columns like 'Price'
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype(target_type)
                message = f"Cast {col} to {target_type}."

            elif tool == "median_impute":
                if self.df[col].dtype == object:
                    # Categorical protection: uses Mode if Agent mis-calls Median
                    fill_val = self.df[col].mode()[0]
                    self.df[col] = self.df[col].fillna(fill_val)
                else:
                    # Numeric logic
                    fill_val = self.df[col].median()
                    self.df[col] = self.df[col].fillna(round(fill_val, 2))
                    
                    # Specific type-casting to match Ground Truth
                    if 'age' in col.lower():
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce').round(0).astype('Int64')
                    elif any(x in col.lower() for x in ['price', 'clicks']):
                        self.df[col] = self.df[col].apply(lambda x: round(float(x), 2) if pd.notnull(x) else x)

                message = f"Median Impute on {col}."

            elif tool == "knn_impute":
                from sklearn.impute import KNNImputer
                temp_series = pd.to_numeric(self.df[col], errors='coerce').values.reshape(-1, 1)
                imputed = KNNImputer(n_neighbors=params.get("n_neighbors", 5)).fit_transform(temp_series)
                self.df[col] = imputed.flatten()
                
                # Correct type-casting for KNN results
                if 'age' in col.lower():
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce').round(0).astype('Int64')
                elif any(x in col.lower() for x in ['price', 'clicks']):
                    self.df[col] = self.df[col].apply(lambda x: round(float(x), 2) if pd.notnull(x) else x)
                
                if any(x in col.lower() for x in ['id', 'idx', 'key']):
                    self.df[col] = self.df[col].apply(lambda x: str(int(float(x))) if pd.notnull(x) else x)
                
                message = f"KNN Impute on {col}."

            # 5. Post-execution updates
            self._update_weights()
            
            # Calculate Reward Components
            bonus = float(calculate_rarity_bonus(self.history, tool))
            gain = calculate_cleaning_gain(old_df, self.df, col, tool, self.weights, self.target_df)
            total_reward = float(gain + bonus)

            # 6. Final Status Checks with explicit casting to Python types
            is_clean = bool(self.df.isnull().sum().sum() == 0)
            is_limit_reached = bool(self.current_step >= self.step_limit)
            done = is_clean or is_limit_reached

            return self._get_observation(message), total_reward, done, {}

        except Exception as e:
            # Safe exit for the API if logic fails
            return self._get_observation(f"Error: {str(e)}"), -1.0, False, {}  


    def _get_observation(self, message: str) -> Observation:
        """Constructs the view seen by the Agent, hiding IDs from the cleaning report."""
        from logic import get_weighted_missing_report
        
        # 1. Calculate the true weighted missingness
        weighted_report = get_weighted_missing_report(self.df, self.weights)
        
        # 2. Identify ID columns to "silence"
        ignore_cols = {'User_ID', 'Product_ID', 'Customer_ID', 'idx', 'key'}
        
        # 3. Construct the clean report but force IDs to 0.0 missingness
        # This tricks the Agent into thinking they are already perfect.
        clean_report = {str(k): (0.0 if any(i in str(k).lower() for i in ['id', 'idx', 'key']) else float(v)) 
                        for k, v in weighted_report.items()}

        return Observation(
            # We keep the preview as-is so the Agent can see row associations
            data_preview=self.df.head().to_dict(orient='records'),
            missing_report=clean_report, 
            schema_info=self.df.dtypes.apply(lambda x: x.name).to_dict(),
            total_rows=len(self.df),
            message=message
        )

    def grader(self) -> float:
        """Final objective score based on ground truth data, ignoring ID columns."""
        print(f"\n--- [GRADER REPORT: {self.task_id.upper()}] ---")
        
        # 1. Identify columns to ignore
        ignore_cols = ['User_ID', 'Product_ID', 'Customer_ID', 'idx', 'key', 'id']
        
        # 2. Hard Task Logic (Governance First)
        if self.task_id == "hard":
            # Check if flag_human was actually called in the history
            governance_passed = "flag_human" in self.history
            score = 1.0 if governance_passed else 0.0
            print(f"Governance Check: {'PASSED' if governance_passed else 'FAILED (No flag_human found)'}")
            return score
        
        # 3. Easy/Medium Task Logic (Data Accuracy)
        # Filter the current and target dataframes to exclude IDs
        cols_to_drop = [c for c in self.df.columns if any(k in c.lower() for k in ignore_cols)]
        current_data = self.df.drop(columns=cols_to_drop)
        target_data = self.target_df.drop(columns=cols_to_drop)
        
        # Reset indices to ensure alignment
        current = current_data.reset_index(drop=True)
        target = target_data.reset_index(drop=True)
        
        total_elements = target.size
        if total_elements == 0: 
            print(f"Grader Error: Data columns not found.")
            return 0.0
        
        # Use .astype(str) to bridge the '1.0' vs '1' gap for numeric data columns like Price
        # and strip any whitespace just in case
        matches = (current.astype(str).apply(lambda x: x.str.strip()) == 
                target.astype(str).apply(lambda x: x.str.strip())).sum().sum()
        
        score = float(matches / total_elements)

        # 4. Diagnostics
        print(f"Total Cells (Data Only): {total_elements}")
        print(f"Matched Cells: {matches}")
        print(f"Mismatched Cells: {total_elements - matches}")
        print(f"Final Accuracy Score: {score:.2%}")

        if score < 1.0:
            # Diagnostic print to see exactly what columns are failing
            mismatched_cols = (current.astype(str) != target.astype(str)).any()
            print(f"Mismatched Columns: {list(mismatched_cols[mismatched_cols].index)}")
            print("Hint: Check for floating point artifacts or rounding in your Imputation tools.")
        
        print("----------------------------------\n")
        
        return score
        