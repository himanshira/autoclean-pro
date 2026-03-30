"""Baseline inference script for AutoClean-Pro OpenEnv.

This script demonstrates a reproducible baseline score by running an AI agent against the 3 task levels (easy, medium, hard).
"""
"""
MANDATORY REQUISITES:
API_BASE_URL: Set via environment
MODEL_NAME: Set via environment
HF_TOKEN: Set via environment
"""

import os 
import json
import re
import requests
from typing import Dict, Any, List, Optional
from openai import OpenAI
from models import Action
from dotenv import load_dotenv

load_dotenv() 

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")# Must use HF_TOKEN as per requirements
MODEL_NAME = os.getenv("MODEL_NAME")
# Use the OpenEnv URL from the environment or default to the Space's own address
OPENENV_URL = os.getenv("OPENENV_URL", "http://localhost:7860") 

# Initialize the client exactly as required
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


SYSTEM_PROMPT = """You are an expert Data Engineering Agent.
Your goal is 100% data cleanliness and strict policy adherence for DATA columns.

[OPERATIONAL POLICY]
1. IDENTIFIERS: Columns like 'User_ID', 'Product_ID', and 'Customer_ID' are handled by the system. You should focus ONLY on columns with missing values in the 'missing_report'.
2. DATA TYPES: 
   - 'Age', 'Clicks', 'Price', 'Income_k' must be numeric.
   - 'Category', 'Survey_Response' are categorical/objects.
3. TERMINATION: If all values in the 'missing_report' are 0.0, your ONLY action must be 'finish'.

[WEIGHTED MISSINGNESS DECISION TREE]
The 'missing_report' uses Bayesian weighting. Follow these thresholds strictly:
- Value >= 0.40 (40%): GOVERNANCE ALERT. You MUST use 'flag_human'. (This will set values to "Review Required").
- 0.15 <= Value < 0.40: Use 'knn_impute'.
- 0.05 <= Value < 0.15: Use 'median_impute'.
- Value < 0.05: Use 'mode_impute' for categories or 'median_impute' for numbers.

[CRITICAL RULES]
- PRICE COLUMN: Even if 'Price' is an 'object', use 'median_impute'. The environment will automatically handle the numeric cast and calculate the median.
- CATEGORY COLUMN: Use 'mode_impute' for missing categorical values.
- REPETITION: If an action does not reduce the missingness in the 'missing_report', do not repeat it. Try a different tool or move to another column.
- FINISH: When the report shows 0.0 for all relevant data columns, you must call 'finish' to submit the task and calculate your accuracy score."""

import json
import re

def get_agent_decision(observation: Dict[str, Any], prev_message: str = "") -> Action:
    # ROunding logic: we round to 3 decimal places to keep precision for Bayesian weights
    # but remove scientiific notation noise like 1e-16
    raw_report = observation.get('missing_report', {})
    clean_report = {k: round(float(v), 3) for k, v in raw_report.items()} 
    prompt = f"""
    CURRENT DATA STATE:
    - Data Preview: {json.dumps(observation.get('data_preview', []), indent=2)}
    - Missingness Report (Weighted): {json.dumps(observation.get('missing_report', {}), indent=2)}
    - Schema Info (Current Types): {json.dumps(observation.get('schema_info', {}), indent=2)}
    
    PREVIOUS FEEDBACK: {prev_message}
    
    TASK: Determine the next best action. 
    If you see a column name like 'User_ID' that is currently 'float64', your FIRST action must be cast_type to 'object'.
    
    Respond ONLY with a JSON object:
    {{
        "tool": "tool_name",
        "column": "column_name",
        "params": {{"key": "value"}}
    }}
    """

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, 
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                temperature=0.0
            )
            
            content = response.choices[0].message.content
            
            # IMPROVED REGEX: Find the first { and the last } 
            # This ignores "Extra data" like comments or second blocks.
            json_match = re.search(r'(\{.*\})', content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1).strip()
                
                # REPAIR: If the LLM outputted two JSON blocks, 
                # json.loads will fail. We take only the FIRST one.
                try:
                    raw_json = json.loads(json_str)
                except json.JSONDecodeError:
                    # If it fails, try to find the first complete object
                    # by looking for the first balanced closing brace
                    depth = 0
                    for i, char in enumerate(json_str):
                        if char == '{': depth += 1
                        elif char == '}': depth -= 1
                        if depth == 0:
                            json_str = json_str[:i+1]
                            break
                    raw_json = json.loads(json_str)

                # --- REPAIR WRAPPER / KEY BUGS ---
                if "actions" in raw_json and isinstance(raw_json["actions"], list):
                    raw_json = raw_json["actions"][0]
                
                valid_tools = ["knn_impute", "median_impute", "mode_impute", "flag_human", "cast_type", "finish"]
                if "tool" not in raw_json:
                    for t in valid_tools:
                        if t in raw_json and isinstance(raw_json[t], dict):
                            nested = raw_json[t]
                            raw_json = {"tool": t, "column": nested.get("column"), "params": nested.get("params", {})}
                            break

                return Action(**raw_json)
            
        except Exception as e:
            print(f"Final Repair Attempt Failed: {e}")
            if attempt == 1: return Action(tool="finish", column="None", params={})

def run_task(task_id: str):
    print(f"\n{'='*15} STARTING TASK: {task_id.upper()} {'='*15}")
    res_data = requests.post(f"{OPENENV_URL}/reset?task_id={task_id}").json()
    obs = res_data.get("observation", {})
    
    # Track the environment message to pass back to the agent for reflection
    last_msg = obs.get("message", "")
    
    step_count = 0
    MAX_STEPS = 15
    
    while step_count < MAX_STEPS:
        step_count += 1
        
        # Pass the last_msg so the agent knows if it failed previously
        # Inside run_task while loop:
        action = get_agent_decision(obs, last_msg)
        print(f"DEBUG: Current Type of {action.column} is {obs.get('schema_info', {}).get(action.column)}")
        
        print(f"Step {step_count}: Executing {action.tool} on '{action.column}'")
        
        res = requests.post(f"{OPENENV_URL}/step?task_id={task_id}", json=action.model_dump()).json()
        
        obs = res.get('observation', {})
        last_msg = obs.get('message', "Action completed.")
        
        if res.get('done', False) or action.tool == "finish":
            print(f"Task finished at step {step_count}. Reason: {last_msg}")
            break

    score_res = requests.get(f"{OPENENV_URL}/grader?task_id={task_id}").json()
    score = score_res.get('score', 0.0)
    print(f"--- {task_id.upper()} FINAL SCORE: {score:.2%} ---")
    return score

if __name__ == "__main__":
    results = {}
    for level in ["easy", "medium", "hard"]:
        results[level] = run_task(level)
    
    print("\n" + "="*40)
    print("FINAL SUMMARY REPORT")
    for lvl, sc in results.items():
        print(f" {lvl.capitalize()}: {sc:.2%}")
    print("="*40)