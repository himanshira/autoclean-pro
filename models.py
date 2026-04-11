from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal

# Strict allow-list keeps the agent on track and prevents hallucinated tool names
ToolType = Literal[
    "knn_impute",
    "median_impute",
    "mode_impute",
    "flag_human",
    "fillna",
    "cast_type",
    "finish",
]


class Action(BaseModel):
    tool: ToolType   # was: tool: str
    column: Optional[str]           = None
    params: Optional[Dict[str, Any]] = {}

    def __init__(self, **data):
        # Silently remap hallucinated field names the LLM sometimes produces
        if "cleaning_tool" in data and "tool" not in data:
            data["tool"] = data.pop("cleaning_tool")
        if "column_to_clean" in data and "column" not in data:
            data["column"] = data.pop("column_to_clean")
        super().__init__(**data)


class Observation(BaseModel):
    # What the agent sees
    data_preview:   List[Dict[str, Any]]
    missing_report: Dict[str, float]      # per-column weighted missingness (0.0–1.0)
    schema_info:    Dict[str, str]        # col → dtype string
    total_rows:     int
    message:        str

    # Bayesian context — tells the LLM which mode is active
    # so it can reason about scores correctly
    weighting_mode:  str = "standard_uniform"   # "bayesian_scarce" | "standard_uniform"
    dataset_regime:  str = "unknown"             # e.g. "scarce_10rows"
    available_tools: List[str] = []              # tool registry snapshot


class Reward(BaseModel):
    value:  float
    reason: str


class State(BaseModel):
    """
    OpenEnv-compatible state model.
    episode_id and step_count are required by the framework contract.
    """
    episode_id:      str
    step_count:      int
    task_id:         str       = ""
    history:         List[str] = []
    bayesian_mode:   str       = "auto"
    scarce_threshold: int      = 50
    weighting_mode:  str       = "unknown"
    dataset_regime:  str       = "unknown"