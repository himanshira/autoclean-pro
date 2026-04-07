from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional, Literal

# Strict list of allowed tools to help the Agent stay on track
ToolType = Literal[
    "knn_impute", 
    "median_impute", 
    "mode_impute", 
    "flag_human", 
    "fillna", 
    "cast_type", 
    "finish"
]
from pydantic import BaseModel
from typing import Optional, Dict, Any

class Action(BaseModel):
    tool: str
    column: Optional[str] = None
    params: Optional[Dict[str, Any]] = {}

    # This magic helper allows the model to accept "cleaning_tool" 
    # and map it to "tool" automatically if the LLM hallucinates.
    def __init__(self, **data):
        if "cleaning_tool" in data and "tool" not in data:
            data["tool"] = data.pop("cleaning_tool")
        if "column_to_clean" in data and "column" not in data:
            data["column"] = data.pop("column_to_clean")
        super().__init__(**data)

class Observation(BaseModel):
    data_preview: List[Dict[str, Any]]
    # Corrected to float to support weighted missingness (0.0 - 1.0)
    missing_report: Dict[str, float] 
    schema_info: Dict[str, str]
    total_rows: int
    message: str

class Reward(BaseModel):
    value: float
    reason: str

class State(BaseModel):
    internal_df_json: str
    step_count: int