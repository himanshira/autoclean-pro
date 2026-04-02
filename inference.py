"""
Final Inference Script for AutoClean-Pro.
Integrated with Weighted Missingness and Self-Correction Reflection.
"""
import os 
import json
import re
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from logic import calculate_cleaning_gain, calculate_rarity_bonus
from openai import OpenAI
from models import Action
from dotenv import load_dotenv

load_dotenv() 

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENENV_URL = os.getenv("OPENENV_URL", "http://localhost:7860")
#IMAGE_NAME = os.getenv("IMAGE_NAME")
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """You are an expert Data Engineering Agent.
Your goal is 100% data cleanliness and strict policy adherence for DATA columns.

[OPERATIONAL POLICY]
1. IDENTIFIERS: Columns like 'User_ID', 'Product_ID' should be cast to 'object' ONLY ONCE if they are numeric. If they are already 'object', move to cleaning.
2. DATA TYPES: 'Age', 'Clicks', 'Price', 'Income_k' must be handled as numeric; 'Category', 'Survey_Response' as categorical.
3. TERMINATION: When 'missing_report' shows 0.0 for all relevant data columns, your ONLY action must be 'finish'.

[WEIGHTED MISSINGNESS DECISION TREE]
The 'missing_report' uses Bayesian weighting. You MUST follow these thresholds:
- Value >= 0.35: CRITICAL GOVERNANCE. Use 'flag_human'. (MANDATORY for high-missingness).
- 0.15 <= Value < 0.35: Use 'knn_impute'.
- 0.05 <= Value < 0.15: Use 'median_impute'.
- Value < 0.05: Use 'mode_impute' for categories or 'median_impute' for numbers.
"""

def get_agent_decision(observation: Dict[str, Any], tool_history: List[str], prev_message: str = "") -> Tuple[str, Action]:
    raw_report = observation.get('missing_report', {})
    clean_report = {k: round(float(v), 3) for k, v in raw_report.items()} 
    schema_info = observation.get('schema_info', {})
    history_str = " -> ".join(tool_history[-5:]) if tool_history else "None"
    prompt = f"""
        CURRENT DATA STATE:
        - Missingness Report (Weighted): {json.dumps(clean_report, indent=2)}
        - Schema Info (Current Types): {json.dumps(schema_info, indent=2)}
        - RECENT HISTORY: {history_str}
        - Previous Feedback: {prev_message}

        TASK: 
        1. Check 'User_ID' or 'Product_ID'. If they are 'float64', your first action is cast_type to 'object'.
        2. Otherwise, find the column with the highest missingness.
        3. Apply Decision Tree: Flag if >= 0.35, KNN if 0.15-0.34, Median if < 0.15.
        4. If previous action resulted in no change, YOU MUST CHOOSE A DIFFERENT COLUMN OR TOOL.
        5. Provide a 'thought' or 'insight' about this data column and why you are choosing this tool.

        CRITICAL RULE: 
        If an action is in the RECENT HISTORY and the missingness report shows no improvement, 
        DO NOT repeat it. Choose a different tool or different column.

        Respond ONLY with a JSON object:
        {{  "thought": "Your explanation here...",
            "tool": "tool_name",
            "column": "column_name",
            "params": {{}}
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
            json_match = re.search(r'(\{.*\})', content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1).strip()
                depth = 0
                for i, char in enumerate(json_str):
                    if char == '{': depth += 1
                    elif char == '}': depth -= 1
                    if depth == 0:
                        json_str = json_str[:i+1]
                        break
                
                raw_json = json.loads(json_str)
                if 'actions' in raw_json and isinstance(raw_json["actions"], list):
                    raw_json = raw_json["actions"][0]
                
                valid_tools = ["knn_impute", "median_impute", "mode_impute", "flag_human", "cast_type", "finish"]
                if "tool" not in raw_json:
                    for t in valid_tools:
                        if t in raw_json and isinstance(raw_json[t], dict):
                            nested = raw_json[t]
                            raw_json["tool"] = t
                            raw_json["column"] = nested.get("column")
                            raw_json["params"] = nested.get("params", {})
                            break
                
                thought = raw_json.get("thought", "Proceeding with cleaning.")
                return thought, Action(
                    tool=raw_json.get("tool"),
                    column=raw_json.get("column", "None"),
                    params=raw_json.get("params", {})
                )
        except Exception as e:
            if attempt == 1:
                return "Error", Action(tool="finish", column="None", params={})
            
    return "Fallback", Action(tool="finish", column="None", params={})

# --- Logging Helpers ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

def run_task(task_id: str):
    log_start(task=task_id, env="autoclean_benchmark", model=MODEL_NAME)
    rewards, tool_history = [], []
    steps_taken, success = 0, False

    try:
        res_data = requests.post(f"{OPENENV_URL}/reset?task_id={task_id}").json()
        obs = res_data.get("observation", {})
        
        for step in range(1, 16):
            # A. CAPTURE OLD STATE (Critical for reward!)
            old_df = pd.DataFrame(obs.get('data_preview', []))

            # B. Get Decision
            thought, action = get_agent_decision(obs, tool_history, obs.get("message", ""))

            # C. Execute Step
            res = requests.post(f"{OPENENV_URL}/step?task_id={task_id}", json=action.model_dump()).json()
            
            # D. Record and Process
            tool_history.append(action.tool)
            obs = res.get('observation', {})
            new_df = pd.DataFrame(obs.get('data_preview', []))
            done = res.get('done', False) or action.tool == "finish"
            
            # E. REWARD CALCULATION
            if action.tool != "finish":
                weights = np.ones(len(new_df))
                try:
                    gain = calculate_cleaning_gain(old_df, new_df, action.column, action.tool, weights)
                except KeyError:
                    gain=0.0    
                rarity = calculate_rarity_bonus(tool_history, action.tool)
                current_step_reward = float((gain + rarity) * 50.0)
            else:
                current_step_reward = 0.0

            rewards.append(current_step_reward)
            steps_taken = step
            verbose_action = f"{thought} | {action.tool}({action.column})"
            
            # Using 'action' as keyword to match your log_step signature
            log_step(step=step, action=verbose_action, reward=current_step_reward, done=done, error=None)
            
            if done: break

        score_res = requests.get(f"{OPENENV_URL}/grader?task_id={task_id}").json()
        final_score = score_res.get('score', 0.0)
        success = final_score >= 0.9
        if rewards: rewards[-1] = final_score

    except Exception as e:
        print(f"!!! CRASH IN RUN_TASK: {e}")
    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

if __name__ == "__main__":
    for level in ["easy", "medium", "hard"]:
        try:
            run_task(level)
        except Exception:
            pass