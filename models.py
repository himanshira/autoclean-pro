"""Data models for the AutoClean-Pro OpenEnv environment."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal

# Use Literal for discrete choices to enable better IDE support and validation
ToolType = Literal["fillna", "drop_rows", "cast_type", "knn_impute", "mode_impute", "flag_human", "finish"]
class Action(BaseModel):
    """Represents a data cleaning operation sent by the agent."""
    tool: ToolType = Field(
        ..., 
        description="The specific cleaning tool to apply to the dataset."
    )
    column: str = Field(
        ..., 
        description="The target column name for the operation."
    )
    params: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional parameters (e.g., {'k': 5} for KNN or {'value': 0} for fillna)."
    )

    class Config:
        """Pydantic configuration for the Action model."""
        json_schema_extra = {
            "example": {
                "tool": "kn_impute",
                "column":"Age",
                "params": {"neighbors": 5}
            }
        }

class Observation(BaseModel):
    """The state of the data visible to the agent."""
    data_preview: List[Dict[str, Any]] = Field(..., description="JSON snapshot of top 5 rows")
    missing_report: Dict[str, int] = Field(..., description="Count of NaNs per column")
    schema_info: Dict[str, str] = Field(..., description="Map of column to data type")
    message: str = Field(..., description="Feedback from the environment")

class Reward(BaseModel):
    """The scalar reward and reasoning for the agent's progress."""
    value: float = Field(..., description="Score between 0.0 and 1.0")
    reason: str = Field(..., description="Explanation for the score and bonuses")

class State(BaseModel):
    """The internal ground-truth state, used for grading."""
    internal_df_json: str = Field(..., description="Full dataframe serialized to JSON")
    step_count: int = Field(..., description="Number of steps taken in current episode")