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

DEFAULT_URL = "https://api-inference.huggingface.co/v1/" 
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_URL)
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL)
HF_TOKEN = os.getenv("HF_TOKEN")
if os.name == 'nt': # 'nt' means Windows
    DEFAULT_OPENENV = "http://127.0.0.1:7860"
else:
    DEFAULT_OPENENV = "http://0.0.0.0:7860"

OPENENV_URL = os.getenv("OPENENV_URL", DEFAULT_OPENENV)
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "autoclean-pro:final")
#IMAGE_NAME = os.getenv("IMAGE_NAME")
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT ="""You are a deterministic Data Engineering Agent. 
Follow these rules in strict order:
[CAST TYPE RULES]
- If a column is an ID column like Product_ID, User_ID check its data type. If the data type of such column is int, int64 or float or float64 convert that column to object datatype.
- If a column consists of discrete or continuous numbers like Age, Price, Income, Sales and such a column has a data type object or category means if it is string then convert it to int64 if is a discrete number like Age and to float64 if it is a continuous number like price, income or sales.
- If a column is categorical but its data type is int or float or int64 or float64, convert that column to object data type.

[IMPUTATION RULES]
- 0.0 < Value < 0.05: Use 'mode_impute' for objects, 'median_impute' for numbers.
- 0.05 <= Value < 0.15: If Numeric, use 'median_impute'.
- 0.15 <= Value < 0.35: If Numeric, use 'knn_impute'.
- Value >= 0.35: CRITICAL. Use 'flag_human'. 
- CATEGORICAL: Any missingness < 0.35 must use 'mode_impute'.  

[TERMINATION RULES] 
If all missingness values are 0.0, use 'finish'.
Note: Do not argue with the schema. If the report says a column has missing values, it needs cleaning regardless of previous thoughts."""

def get_agent_decision(observation: Dict[str, Any], tool_history: List[str], prev_message: str = "") -> Tuple[str, Action]:
    raw_report = observation.get('missing_report', {})
    clean_report = {k: round(float(v), 3) for k, v in raw_report.items()} 
    schema_info = observation.get('schema_info', {})
    history_str = " -> ".join(tool_history[-5:]) if tool_history else "None"
    prompt = f"""
        OBSERVATION:
        - Missingness Report (Weighted): {json.dumps(clean_report, indent=2)}
        - Schema Info (Current Types): {json.dumps(schema_info, indent=2)}
        - Last Action: {tool_history[-1] if tool_history else "None"}

        INSTRUCTION:
        1. Identify the highest missingness column.
        2. Select the tool based on the [STRICT TOOL-TYPE MAPPING].
        3. If the 'Last Action' failed to reduce missingness, try a different tool for that same column.


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
            
            # --- IMPROVED EXTRACTION LOGIC ---
            # 1. Use regex to find the FIRST curly brace to the LAST curly brace
            json_match = re.search(r'(\{.*\})', content, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in LLM response")
            
            json_str = json_match.group(1).strip()
            
            # 2. Basic JSON Load
            raw_json = json.loads(json_str)
            
            # 3. Handle 'actions' wrapper if LLM hallucinated a list
            if isinstance(raw_json, dict) and "actions" in raw_json:
                raw_json = raw_json["actions"][0] if isinstance(raw_json["actions"], list) else raw_json["actions"]
            
            # 4. Standardize keys (sometimes LLMs use 'column_name' instead of 'column')
            tool = raw_json.get("tool") or raw_json.get("action")
            column = raw_json.get("column") or raw_json.get("column_name", "None")
            params = raw_json.get("params") or {}
            thought = raw_json.get("thought", "Cleaning step...")

            # Validate tool
            valid_tools = ["knn_impute", "median_impute", "mode_impute", "flag_human", "cast_type", "finish"]
            # If the tool name is missing but the key is the tool name itself
            if tool not in valid_tools:
                 for t in valid_tools:
                     if t in raw_json:
                         tool = t
                         nested = raw_json[t]
                         if isinstance(nested, dict):
                            column = nested.get("column", column)
                            params = nested.get("params", params)
                            # Grab the thought from nested if it's missing from root
                            thought = nested.get("thought", thought)
                         break
            tool = str(tool).lower() if tool else "finish"
            return thought, Action(tool=tool, column=column, params=params)

        except Exception as e:
            print(f"DEBUG: Attempt {attempt+1} failed: {e}")
            if attempt == 1:
                return "Error Recovery", Action(tool="finish", column="None", params={})
            
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