"""
AutoClean-Pro environment — v2

Architecture
------------
BaseCleanEnv   Abstract base class. Defines the contract:
               reset(), step(), state, close().
               Knows nothing about column names, dataset paths,
               or task identity — those are implementation details.

ToolRegistry   Decorator-based registry. Tools register themselves
               with @ToolRegistry.register("name"). The environment
               discovers them at runtime from the loaded dataframe —
               no hardcoded column lists anywhere.

AutoCleanEnv   Concrete implementation. Inherits BaseCleanEnv,
               uses ToolRegistry for dispatch.
               Bayesian weighting mode is a constructor parameter,
               not a hardcoded behaviour.
"""

import sys
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from models import Action, Observation, State
from logic import (
    get_weighted_missing_report,
    calculate_cleaning_gain,
    calculate_rarity_bonus,
    validate_cleaning_strategy,
)


# ===========================================================================
# Tool Registry
# ===========================================================================

class ToolRegistry:
    """
    Decorator-based registry for cleaning tools.

    Usage
    -----
    @ToolRegistry.register("median_impute")
    def _median_impute(df, col, params):
        ...

    The environment calls ToolRegistry.execute(tool, df, col, params)
    without knowing column names in advance.
    """
    _tools: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(fn: Callable) -> Callable:
            cls._tools[name] = fn
            return fn
        return decorator

    @classmethod
    def execute(cls, tool: str, df: pd.DataFrame,
                col: Optional[str], params: Dict) -> Tuple[pd.DataFrame, str]:
        """
        Dispatch tool call. Returns (mutated_df, message).
        Raises KeyError for unknown tools.
        """
        if tool not in cls._tools:
            raise KeyError(f"Unknown tool: '{tool}'")
        return cls._tools[tool](df, col, params)

    @classmethod
    def available(cls) -> List[str]:
        return sorted(cls._tools.keys())


# ---------------------------------------------------------------------------
# Tool implementations — registered at module load time.
# None of these reference column names directly; they receive `col` at runtime.
# ---------------------------------------------------------------------------

@ToolRegistry.register("flag_human")
def _flag_human(df: pd.DataFrame, col: str, params: Dict):
    return df.copy(), f"Column '{col}' flagged for human review."


@ToolRegistry.register("cast_type")
def _cast_type(df: pd.DataFrame, col: str, params: Dict):
    target = params.get("target_dtype", "float64")
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce").astype(target)
    return df, f"Cast '{col}' to {target}."


@ToolRegistry.register("median_impute")
def _median_impute(df: pd.DataFrame, col: str, params: Dict):
    df = df.copy()
    # Guard: if column is object/str and can't be parsed as numeric,
    # redirect to mode_impute to prevent destroying the column with NaN.
    if df[col].dtype == object or str(df[col].dtype) in ("str","string"):
        modes = df[col].mode()
        fill = modes.iloc[0] if not modes.empty else ""
        df.loc[:, col] = df[col].fillna(fill)
        return df, (
            f"median_impute redirected to mode_impute on object column '{col}'. "
            f"Use mode_impute directly for categorical columns."
        )
    s = pd.to_numeric(df[col], errors="coerce")
    median_val = s.median()
    if pd.isna(median_val):
        # All values unparseable — do nothing rather than destroy the column
        return df, f"median_impute skipped '{col}': no parseable numeric values."
    df.loc[:, col] = s.fillna(median_val)
    return df, f"Median impute on '{col}' (median={median_val:.4g})."


@ToolRegistry.register("mode_impute")
def _mode_impute(df: pd.DataFrame, col: str, params: Dict):
    df = df.copy()
    modes = df[col].mode()
    fill = modes.iloc[0] if not modes.empty else ""
    df.loc[:, col] = df[col].fillna(fill)
    return df, f"Mode impute on '{col}'."


@ToolRegistry.register("knn_impute")
def _knn_impute(df: pd.DataFrame, col: str, params: Dict):
    df = df.copy()
    numeric = pd.to_numeric(df[col], errors="coerce")
    # Redirect categorical columns to mode_impute — KNNImputer is numeric only
    if df[col].dtype == object or numeric.isna().sum() == len(df):
        modes = df[col].mode()
        fill = modes.iloc[0] if not modes.empty else ""
        df.loc[:, col] = df[col].fillna(fill)
        return df, f"knn_impute redirected to mode_impute on categorical '{col}'."
    k = int(params.get("n_neighbors", 5))
    vals = numeric.values.reshape(-1, 1)
    imputed = KNNImputer(n_neighbors=k).fit_transform(vals).flatten()
    df.loc[:, col] = imputed
    return df, f"KNN impute (k={k}) on '{col}'."


@ToolRegistry.register("multifeature_knn_impute")
def _multifeature_knn_impute(df: pd.DataFrame, col: str, params: Dict):
    """
    Multi-feature KNN imputation for large datasets.
    Uses ALL numeric columns as neighbours, not just the target column.
    This gives contextually accurate fills: e.g. Salary predicted from
    AtBat, Hits, Years, etc. rather than just from other Salary values.

    Recommended when n > 50 rows and multiple numeric features exist.
    Falls back to single-column knn_impute if only one numeric column present.
    """
    df = df.copy()

    # Collect all numeric columns (excluding the target temporarily)
    numeric_cols = [
        c for c in df.columns
        if np.issubdtype(df[c].dtype, np.number) or
           pd.to_numeric(df[c], errors="coerce").notna().any()
    ]

    if len(numeric_cols) < 2:
        # Only one numeric column — fall back to single-column KNN
        s = pd.to_numeric(df[col], errors="coerce")
        vals = s.values.reshape(-1, 1)
        imputed = KNNImputer(n_neighbors=min(5, int(s.notna().sum()))).fit_transform(vals)
        df.loc[:, col] = imputed.flatten()
        return df, f"multifeature_knn_impute fell back to single-column KNN on '{col}' (only 1 numeric col)."

    # Build numeric matrix from all numeric columns
    num_df = pd.DataFrame({
        c: pd.to_numeric(df[c], errors="coerce") for c in numeric_cols
    })

    k = int(params.get("n_neighbors", 5))
    k = min(k, int(num_df[col].notna().sum()))  # never exceed available rows

    imputed_matrix = KNNImputer(n_neighbors=k).fit_transform(num_df)
    imputed_df = pd.DataFrame(imputed_matrix, columns=numeric_cols, index=df.index)

    # Only update the target column
    df.loc[:, col] = imputed_df[col]
    n_features = len(numeric_cols)
    return df, (
        f"Multi-feature KNN (k={k}, {n_features} features) imputed '{col}'. "
        f"Features used: {numeric_cols[:5]}{'...' if n_features > 5 else ''}."
    )


@ToolRegistry.register("fillna")
def _fillna(df: pd.DataFrame, col: str, params: Dict):
    df = df.copy()
    val = params.get("value", 0)
    df.loc[:, col] = df[col].fillna(val)
    return df, f"fillna '{col}' with {val}."


@ToolRegistry.register("finish")
def _finish(df: pd.DataFrame, col: Optional[str], params: Dict):
    return df.copy(), "Agent submitted dataset for grading."


# ===========================================================================
# State dataclass (OpenEnv-style)
# ===========================================================================

@dataclass
class EpisodeState:
    """
    Separates internal metadata from the agent's observation.
    Matches OpenEnv's State contract: episode_id + step_count required.
    """
    episode_id:     str       = field(default_factory=lambda: str(uuid.uuid4()))
    step_count:     int       = 0
    task_id:        str       = ""
    history:        List[str] = field(default_factory=list)
    bayesian_mode:  str       = "auto"
    scarce_threshold: int     = 50
    weighting_mode: str       = "unknown"   # resolved on reset
    dataset_regime: str       = "unknown"   # resolved on reset
    flagged_cols:   List[str] = field(default_factory=list)  # columns handled by flag_human


# ===========================================================================
# Abstract base
# ===========================================================================

class BaseCleanEnv(ABC):
    """
    Abstract base for all cleaning environments.

    Subclasses must implement:
        reset()  → Observation
        step()   → Dict[str, Any]
        grader() → float
        close()

    The base class knows nothing about file paths, column names,
    or Bayesian parameters — those are concrete concerns.
    """

    @abstractmethod
    def reset(self) -> Observation:
        ...

    @abstractmethod
    def step(self, action: Action) -> Dict[str, Any]:
        ...

    @abstractmethod
    def grader(self, silent: bool = False) -> float:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @property
    @abstractmethod
    def state(self) -> EpisodeState:
        ...


# ===========================================================================
# Concrete environment
# ===========================================================================

class AutoCleanEnv(BaseCleanEnv):
    """
    Concrete data-cleaning RL environment.

    Key design decisions
    --------------------
    - Column names are NEVER hardcoded. The environment discovers
      them from the loaded dataframe at reset() time.
    - Tool dispatch goes through ToolRegistry — adding a new cleaning
      tool requires only decorating a function, not editing this class.
    - bayesian_mode is a constructor parameter ("auto"|"on"|"off"),
      making Bayesian weighting a tunable feature rather than fixed logic.
    - The grader uses np.isclose with rtol=0.02 to tolerate
      hardware-level float variance (CPU vs TPU imputation differences).
    """

    SUPPORTS_CONCURRENT_SESSIONS = False   # OpenEnv contract flag

    def __init__(
        self,
        task_id: str,
        step_limit: int = 15,
        bayesian_mode: str = "auto",   # "auto" | "on" | "off"
        scarce_threshold: int = 50,    # rows ≤ this → scarce dataset
    ):
        self.task_id         = task_id
        self.step_limit      = step_limit
        self._bayesian_mode  = bayesian_mode
        self._scarce_threshold = scarce_threshold

        self.df:        pd.DataFrame = None
        self.target_df: pd.DataFrame = None
        self.weights:   np.ndarray   = None

        self._state = EpisodeState(
            task_id=task_id,
            bayesian_mode=bayesian_mode,
            scarce_threshold=scarce_threshold,
        )

        self._paths = {
            "easy":   {"dirty": "data/easy_dirty.csv",  "clean": "data/easy_clean.csv"},
            "medium": {"dirty": "data/med_dirty.csv",   "clean": "data/med_clean.csv"},
            "hard":   {"dirty": "data/hard_dirty.csv",  "clean": "data/hard_clean.csv"},
        }

        self.reset()

    # ------------------------------------------------------------------
    # OpenEnv contract: state property
    # ------------------------------------------------------------------

    @property
    def state(self) -> EpisodeState:
        return self._state

    def get_state_model(self) -> State:
        """
        Returns models.State — the OpenEnv-spec compliant Pydantic model.
        Used by the /state HTTP endpoint in app.py.
        """
        s = self._state
        return State(
            episode_id       = s.episode_id,
            step_count       = s.step_count,
            task_id          = s.task_id,
            history          = s.history,
            bayesian_mode    = s.bayesian_mode,
            scarce_threshold = s.scarce_threshold,
            weighting_mode   = s.weighting_mode,
            dataset_regime   = s.dataset_regime,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        pd.set_option("future.no_silent_downcasting", True)

        paths = self._paths[self.task_id]
        self.df = pd.read_csv(
            paths["dirty"],
            na_values=["", " ", "nan", "NaN", "None", "null", "nan.0", "Nan"],
            keep_default_na=True,
        )
        self.target_df = pd.read_csv(paths["clean"])

        # Discover dataset regime — no hardcoded column names
        n = len(self.df)
        is_scarce = (n <= self._scarce_threshold)
        weighting_mode = (
            "bayesian_scarce" if (
                self._bayesian_mode == "on" or
                (self._bayesian_mode == "auto" and is_scarce)
            ) else "standard_uniform"
        )

        self._state = EpisodeState(
            episode_id      = str(uuid.uuid4()),
            step_count      = 0,
            task_id         = self.task_id,
            history         = [],
            bayesian_mode   = self._bayesian_mode,
            scarce_threshold= self._scarce_threshold,
            weighting_mode  = weighting_mode,
            dataset_regime  = f"{'scarce' if is_scarce else 'standard'}_{n}rows",
        )

        self._update_weights()
        return self._get_observation("Environment reset.")

    def step(self, action: Action) -> Dict[str, Any]:
        self._state.step_count += 1
        col    = action.column
        tool   = action.tool
        params = action.params or {}

        self._state.history.append(tool)
        if tool == "flag_human" and col and col not in self._state.flagged_cols:
            self._state.flagged_cols.append(col)

        # Finish or invalid column
        if tool == "finish" or col is None or col not in self.df.columns:
            is_clean   = bool(int(self.df.isnull().sum().sum()) == 0)
            is_success = self._is_success_by_state(is_clean)
            return self._json_safe({
                "observation": self._get_observation("Task finalised."),
                "reward":      0.0,
                "done":        True,
                "info":        {"success": is_success, "last_action_error": "null"},
                "success":     is_success,
            })

        old_df = self.df.copy()

        try:
            # Dispatch via ToolRegistry — environment never names tools directly
            new_df, message = ToolRegistry.execute(tool, self.df, col, params)
            self.df = new_df

            self._update_weights()

            bonus = float(calculate_rarity_bonus(self._state.history, tool))
            gain  = float(calculate_cleaning_gain(
                old_df, self.df, col, tool, self.weights, self.target_df
            ))
            total_reward = gain + bonus

            is_clean   = bool(int(self.df.isnull().sum().sum()) == 0)
            done       = is_clean or (self._state.step_count >= self.step_limit)
            is_success = self._is_success_by_state(is_clean)

            return self._json_safe({
                "observation": self._get_observation(message),
                "reward":      total_reward,
                "done":        done,
                "info":        {"success": is_success, "last_action_error": "null"},
                "success":     is_success,
            })

        except KeyError:
            # Unknown tool name
            msg = f"Unknown tool '{tool}'. Available: {ToolRegistry.available()}"
            return self._json_safe({
                "observation": self._get_observation(msg),
                "reward":      -0.5,
                "done":        False,
                "info":        {"success": False, "last_action_error": msg},
                "success":     False,
            })

        except Exception as e:
            return self._json_safe({
                "observation": self._get_observation(f"Error: {e}"),
                "reward":      -1.0,
                "done":        False,
                "info":        {"success": False, "last_action_error": str(e)},
                "success":     False,
            })

    def grader(self, silent: bool = False) -> float:
        """
        Score the current state against ground truth.
        silent=True suppresses all output (used by inference.py in finally).
        Scores are clamped to (0.001, 0.999) — strictly between 0 and 1
        as required by the Phase 2 validator.

        For custom uploaded CSVs (no ground-truth clean file):
        Scores by NaN elimination — what fraction of originally dirty
        cells have been filled. This is self-contained and works for
        any user-uploaded dataset without needing a reference file.
        """
        def _clamp(s: float) -> float:
            return max(0.001, min(0.999, float(s)))

        ignore = {"user_id", "product_id", "customer_id", "idx", "key", "id"}

        # Custom uploaded CSV — no ground-truth available.
        # Score = fraction of originally-missing cells now filled.
        if self.task_id == "custom":
            if self.df is None:
                return 0.001
            try:
                # Count cells still missing (governance columns excluded)
                drop = [c for c in self.df.columns
                        if any(k in c.lower() for k in ignore)]
                working = self.df.drop(columns=drop)
                # Also exclude columns flagged for human review
                flagged = self._state.flagged_cols
                working = working.drop(
                    columns=[c for c in flagged if c in working.columns],
                    errors="ignore"
                )
                total_cells   = working.size
                missing_cells = int(working.isnull().sum().sum())
                # Count sentinel strings as still missing
                for col in working.columns:
                    if str(working[col].dtype) in ("object","str","string"):
                        missing_cells += int(working[col].dropna().apply(
                            lambda x: str(x).strip().lower() in
                            {"missing","n/a","na","none","null","unknown","?","-",""}
                        ).sum())
                filled_fraction = 1.0 - (missing_cells / total_cells) if total_cells > 0 else 1.0
                score = _clamp(filled_fraction)
                if not silent:
                    sys.stderr.write(
                        f"[GRADER:CUSTOM] {score:.4f} "
                        f"({missing_cells} cells still missing / {total_cells} total)\n"
                    )
                    sys.stderr.flush()
                return score
            except Exception as e:
                if not silent:
                    sys.stderr.write(f"[GRADER ERROR custom] {e}\n")
                    sys.stderr.flush()
                return 0.001

        if self.task_id == "hard":
            # Partial credit grader:
            # 60% for governance (flag_human on Survey_Response >= 35% missing)
            # 40% for cleaning Income_k (20% missing — imputable)
            flagged = "flag_human" in self._state.history
            # Check if Income_k has been cleaned (no NaNs remaining)
            income_col = next(
                (c for c in (self.df.columns if self.df is not None else [])
                 if "income" in c.lower()),
                None
            )
            income_clean = (
                income_col is not None
                and self.df is not None
                and int(self.df[income_col].isnull().sum()) == 0
            )
            score = (0.6 if flagged else 0.0) + (0.4 if income_clean else 0.0)
            return _clamp(score if score > 0 else 0.001)

        try:
            drop = [c for c in self.df.columns
                    if any(k in c.lower() for k in ignore)]
            cur = self.df.drop(columns=drop).apply(pd.to_numeric, errors="coerce")
            tgt = self.target_df.drop(columns=drop).apply(pd.to_numeric, errors="coerce")

            matches = int(np.sum(
                np.isclose(cur.values, tgt.values, equal_nan=True, rtol=0.02)
            ))
            score = float(matches / int(tgt.size)) if tgt.size > 0 else 0.0
            clamped = _clamp(score)

            if not silent:
                sys.stderr.write(
                    f"[GRADER:{self.task_id.upper()}] {clamped:.4f} "
                    f"({clamped:.2%}) | mode={self._state.weighting_mode}\n"
                )
                sys.stderr.flush()
            return clamped

        except Exception as e:
            if not silent:
                sys.stderr.write(f"[GRADER ERROR] {e}\n")
                sys.stderr.flush()
            return 0.001

    def close(self) -> None:
        """Release resources. Safe to call multiple times."""
        self.df        = None
        self.target_df = None
        self.weights   = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_success_by_state(self, is_clean: bool) -> bool:
        """Fast in-step check — no grader call, no I/O."""
        if self.task_id == "hard":
            return bool("flag_human" in self._state.history)
        return is_clean

    def _update_weights(self) -> None:
        """
        Weights array kept as np.ndarray for logic.py API compatibility.
        The actual missingness computation happens per-column in logic.py.
        """
        n = len(self.df)
        # Simple uniform array — the Bayesian amplification happens in
        # get_weighted_missing_report, not here
        self.weights = np.ones(n, dtype=float)

    def _get_observation(self, message: str) -> Observation:
        """
        Build observation. Column names are discovered dynamically
        from self.df — never hardcoded anywhere in this method.
        """
        report = get_weighted_missing_report(
            self.df,
            self.weights,
            bayesian_mode=self._bayesian_mode,
            scarce_threshold=self._scarce_threshold,
        )

        # Zero out columns already handled by flag_human so the agent
        # doesn't loop on them — flag_human doesn't change the data,
        # so without this the column stays dirty and agent loops forever.
        for flagged in self._state.flagged_cols:
            if flagged in report:
                report[flagged] = 0.0

        return Observation(
            data_preview    = self.df.head().to_dict(orient="records"),
            missing_report  = {str(k): float(v) for k, v in report.items()},
            schema_info     = self.df.dtypes.apply(lambda x: x.name).to_dict(),
            total_rows      = len(self.df),
            message         = message,
            weighting_mode  = self._state.weighting_mode,
            dataset_regime  = self._state.dataset_regime,
            available_tools = ToolRegistry.available(),
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