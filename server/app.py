import os
import sys
import math
import uvicorn
import subprocess
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# Ensure the root directory is on the path (handles running from a subdirectory)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Action, Observation
from environment import AutoCleanEnv

app = FastAPI(title="AutoClean-Pro API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Environment registry — populated lazily on first /reset call
# ---------------------------------------------------------------------------
VALID_TASKS = {"easy", "medium", "hard"}
envs: dict = {}


def _get_env(task_id: str) -> AutoCleanEnv:
    if task_id not in envs:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' not initialised. POST /reset?task_id={task_id} first."
        )
    return envs[task_id]


def _sanitise(obj):
    """
    Recursively replace NaN / Inf floats with JSON-safe values.
    Python's json module raises ValueError on NaN/Inf — this prevents
    the 500 'Out of range float values are not JSON compliant' error.
    """
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    # Pydantic model → dict first, then sanitise
    if hasattr(obj, "model_dump"):
        return _sanitise(obj.model_dump())
    if hasattr(obj, "dict"):
        return _sanitise(obj.dict())
    return obj


# ---------------------------------------------------------------------------
# Cached baseline scores
# ---------------------------------------------------------------------------
BASELINE_RESULTS = {
    "easy":   2.50,
    "medium": 2.85,
    "hard":   4.10,
    "model":  "Qwen/Qwen2.5-7B-Instruct",
    "status": "Verified",
    "date":   "2026-04-07",
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"message": "AutoClean-Pro OpenEnv is Live."}


@app.post("/reset")
async def reset(task_id: str = "easy"):
    if task_id not in VALID_TASKS:
        raise HTTPException(status_code=404, detail=f"Unknown task '{task_id}'.")

    envs[task_id] = AutoCleanEnv(task_id=task_id)
    obs = envs[task_id].reset()

    # _sanitise converts the Observation model to dict AND replaces NaN/Inf
    return {"observation": _sanitise(obs), "info": {}}


@app.post("/step")
async def step(action: Action, task_id: str = "easy"):
    env    = _get_env(task_id)
    result = env.step(action)
    # result comes through env._json_safe already, but _sanitise catches
    # any remaining NaN/Inf that slipped through (e.g. in data_preview)
    return _sanitise(result)


@app.get("/grader")
async def get_grader(task_id: str = "easy"):
    env   = _get_env(task_id)
    score = env.grader(silent=True)
    return _sanitise({
        "task_id": task_id,
        "score":   score,
        "success": (score >= 1.0) if task_id == "hard" else (score > 0.98),
    })


@app.post("/baseline")
async def trigger_baseline(background_tasks: BackgroundTasks):
    def run_inference():
        try:
            proc = subprocess.run(
                ["uv", "run", "python", "inference.py"],
                text=True,
                capture_output=True,
                cwd="/app",
            )
            if proc.stdout:
                sys.stderr.write("[INFERENCE STDOUT]\n" + proc.stdout)
                sys.stderr.flush()
            if proc.stderr:
                sys.stderr.write("[INFERENCE STDERR]\n" + proc.stderr)
                sys.stderr.flush()
            if proc.returncode != 0:
                sys.stderr.write(f"[INFERENCE] exited with code {proc.returncode}\n")
                sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"[BASELINE CRASH] {e}\n")
            sys.stderr.flush()

    background_tasks.add_task(run_inference)
    return {
        "status":          "Inference started in background.",
        "instruction":     "Monitor the Logs tab for [START] / [STEP] / [END] output.",
        "cached_baseline": BASELINE_RESULTS,
    }


@app.get("/baseline")
async def get_baseline():
    return BASELINE_RESULTS


@app.get("/state")
async def get_state(task_id: str = "easy"):
    env = _get_env(task_id)
    return {
        "task_id":       task_id,
        "current_step":  env.current_step,
        "step_limit":    env.step_limit,
        "history":       env.history,
        "missing_total": int(env.df.isnull().sum().sum()),
        "total_rows":    len(env.df),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()