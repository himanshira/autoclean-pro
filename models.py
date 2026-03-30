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

class Action(BaseModel):
    # Using Field with default_factory for params makes it more robust
    tool: ToolType = Field(..., description="The cleaning tool to use")
    column: str = Field(..., description="The target column name")
    params: Dict[str, Any] = Field(default_factory=dict)

    # Allow the model to accept data even if the LLM adds extra fields
    model_config = ConfigDict(
        populate_by_name=True,
        extra='ignore' 
    )

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