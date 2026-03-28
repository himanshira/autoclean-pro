"""Baseline inference script for AutoClean-Pro OpenEnv.

This script demonstrates a reproducible baseline score by running an AI agent against the 3 task levels (easy, medium, hard).
"""

import os 
import json
import openai
from typing import Dict, Any
from models import Action
from dotenv import load_dotenv

load_dotenv() # This looks for the .env file and loads the key
# Setup Client (Reads from environment variables as per requirements)
client = openai.OpenAI(api_key=os.getenv("OPEN_API_KEY"))

def get_agent_decision(observation: Dict[str, Any]) -> Action:
    """Consults the LLM to decide the next cleaning step.
    
    Args:
        observation: The current state summary from the environment.
    Returns:
        A validated Action object.    
    """
    prompt = f"""
    You are a Data Governance Agent. 
    STATUS: {observation['missing_report']}

    RULES:
    1. If a column has 0 NaNs, DO NOT TOUCH IT. Move to the next column.
    2. If a column has >= 50% missing values (e.g., 4 or more), you MUST use 'flag_human'. 
    3. Use 'knn_impute' ONLY for numeric columns with < 50% missing values.
    4. Only use 'finish' when EVERY column in the STATUS shows 0.
    
    Current Goal: Scan STATUS, apply rules, and output JSON.
    
    JSON ONLY: {{"tool": "...", "column": "...", "params": {{}} }}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[{"role": "system", "content": "You are a precise data engineering agent."},
                  {"role": "user", "content": prompt}],
        response_format = {"type": "json_object"}          
    )

    decision = json.loads(response.choices[0].message.content)
    return Action(**decision)

def run_task(task_id: str):
    """Runs a full episode for a specific task difficulty."""
    print(f"--- Starting Task: {task_id.upper()} ---")

    # In a real OpenEnv, you'd call your local server endpoints here
    # For the baseline script, we can interact with the class directly or via requests
    import requests
    base_url = os.getenv(
    "OPENENV_URL", 
    "https://himanshirawat0892-autoclean-pro.hf.space")

    # 1. Reset
    obs = requests.post(f"{base_url}/reset?task_id={task_id}").json()
    done = False
    total_reward = 0

    while not done:
        # Calculate percentage for logs
        report = obs['missing_report']
        total_rows = len(obs['data_preview'])
        status_str = ", ".join([f"{col}: {count} NaN" for col, count in report.items()])
        print(f"Current Status: {status_str}")
        # 2. Agent Decides
        action = get_agent_decision(obs)
        print(f"Agent Action: {action.tool} on {action.column}")

        # 3. Step 
        # Replace your #3 Step block in baseline.py with this:
        response = requests.post(
            f"{base_url}/step?task_id={task_id}",
            json=action.model_dump()
        )

        if response.status_code != 200:
            print(f"SERVER ERROR: {response.text}")
            break # Stop the loop if the server crashed

        data = response.json()
        obs = data['observation']
        #total_reward += data['reward']
        done = data['done']

    # 4. Final Score (Grader)
    final_score = requests.get(f"{base_url}/grader?task_id={task_id}").json()
    print(f"Result: {final_score['score']:.2%}")   
    return final_score['score']

if __name__=="__main__":
    # Ensure server is running before executing this
    scores = {}
    for level in ["easy", "medium", "hard"]:
        scores[level] = run_task(level)
    print("\n--- FINAL REPRODUCIBLE SCORES ---")
    print(json.dumps(scores, indent=2))
         