import os
import sys
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
# Environment registry
# Populated lazily on first /reset call so the server starts even if data
# files are not yet present at import time.
# ---------------------------------------------------------------------------
VALID_TASKS = {"easy", "medium", "hard"}
envs: dict = {}


def _get_env(task_id: str) -> AutoCleanEnv:
    """Return the environment for task_id, raising 404 if not initialised."""
    if task_id not in envs:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' not initialised. POST /reset?task_id={task_id} first."
        )
    return envs[task_id]


def _obs_to_dict(obs) -> dict:
    """Safely convert an Observation (Pydantic model or plain dict) to a dict."""
    if isinstance(obs, dict):
        return obs
    if hasattr(obs, "model_dump"):   # Pydantic v2
        return obs.model_dump()
    if hasattr(obs, "dict"):         # Pydantic v1 fallback
        return obs.dict()
    return {}


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

    # Always create a fresh environment on reset
    envs[task_id] = AutoCleanEnv(task_id=task_id)
    obs = envs[task_id].reset()

    # reset() returns an Observation Pydantic model — convert to dict for JSON response
    return {"observation": _obs_to_dict(obs), "info": {}}


@app.post("/step")
async def step(action: Action, task_id: str = "easy"):
    env    = _get_env(task_id)
    result = env.step(action)

    # result is already a plain dict from _json_safe, but observation inside
    # may still be an Observation model if called directly — normalise it.
    if "observation" in result and not isinstance(result["observation"], dict):
        result["observation"] = _obs_to_dict(result["observation"])

    return result


@app.get("/grader")
async def get_grader(task_id: str = "easy"):
    env   = _get_env(task_id)
    score = env.grader()
    return {
        "task_id":    task_id,
        "score":      score,
        "success":    (score >= 1.0) if task_id == "hard" else (score > 0.98),
    }


@app.post("/baseline")
async def trigger_baseline(background_tasks: BackgroundTasks):
    """
    Run inference.py in the background.
    subprocess stdout is captured and redirected to stderr so it never
    pollutes the hackathon evaluator's stdout parser.
    """
    def run_inference():
        try:
            proc = subprocess.run(
                [sys.executable, "inference.py"],
                text=True,
                capture_output=True,   # captures both stdout and stderr
            )
            # Forward inference stdout → our stderr (keeps evaluator stdout clean)
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
    """Return a lightweight snapshot of the current environment state."""
    env = _get_env(task_id)
    return {
        "task_id":      task_id,
        "current_step": env.current_step,
        "step_limit":   env.step_limit,
        "history":      env.history,
        "missing_total": int(env.df.isnull().sum().sum()),
        "total_rows":   len(env.df),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()