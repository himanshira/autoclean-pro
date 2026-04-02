import os
import sys
import uvicorn
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import Action, Observation, State
from fastapi import BackgroundTasks

# This ensures the 'server' folder can see 'environment.py' in the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import AutoCleanEnv

app = FastAPI(title="AutoClean-Pro API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

envs = {
    "easy": AutoCleanEnv(task_id="easy"),
    "medium": AutoCleanEnv(task_id="medium"),
    "hard": AutoCleanEnv(task_id="hard")
}

BASELINE_RESULTS = {
    "easy": 0.8667,
    "medium": 0.8667,
    "hard": 1.0,
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "status": "In-Progress",
    "date": "2026-04-02"
}

@app.get("/")
async def root():
    return {"message": "AutoClean-Pro OpenEnv is Live."}

@app.post("/reset")
async def reset(task_id: str = "easy"):
    if task_id not in envs:
        raise HTTPException(status_code=404, detail="Task not found")
    
    envs[task_id] = AutoCleanEnv(task_id=task_id)
    
    # Call reset and capture whatever it returns
    result = envs[task_id].reset()
    
    # If result is a tuple (obs, info), unpack it. 
    # If it's just the observation, set info to empty dict.
    if isinstance(result, tuple):
        obs, info = result[0], result[1]
    else:
        obs, info = result, {}

    return {"observation": obs, "info": info}

@app.post("/step")
async def step(action: Action, task_id: str = "easy"):
    if task_id not in envs:
        raise HTTPException(status_code=404, detail="Task not found")
    current_env = envs[task_id]
    
    #This call already uses the updated environment logic 
    # which calculates the Bayesian weighted report.
    obs, reward, done, info = current_env.step(action) 

    # Return only the what the environment gave us
    # Don't manually overwrite the missing_report     
    return {"observation": obs, "reward": reward, "done": done, "info": info}

@app.get("/grader")
async def get_grader(task_id: str = "easy"):
    if task_id not in envs:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"score": envs[task_id].grader()}

@app.post("/baseline")
async def trigger_baseline(background_tasks: BackgroundTasks):
    """REQUIRED BY VALIDATOR: Runs the inference.py script to prove the scores are reproducible."""
    def run_inference():
        try:
            # We use Popen or run without capture_output so logs 
            # flow directly to the HUgging Face Container Logs
            subprocess.run(
                [sys.executable, "inference.py"],
                text=True,
                check=True
            )
        except Exception as e:
            print(f"!!! BACKGROUND CRASH: {str(e)}")
    background_tasks.add_task(run_inference)
    return {
        "status": "Inference started in background",
        "instruction": "Please monitor the 'Logs' tab in your Hugging Face Space for [START] and [END] tags.",
        "cached_baseline": BASELINE_RESULTS
    }        
@app.get("/baseline")
async def get_baseline():
    """Returns the cached results for quick status checks."""
    return BASELINE_RESULTS

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__=="__main__":
    main()