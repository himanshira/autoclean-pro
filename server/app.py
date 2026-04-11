import os
import sys
import math
import uvicorn
import subprocess
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import StreamingResponse
import io
import tempfile
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Action, Observation, State
from environment import AutoCleanEnv, ToolRegistry

app = FastAPI(title="AutoClean-Pro API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Environment registry — lazily populated on /reset
# ---------------------------------------------------------------------------
VALID_TASKS = {"easy", "medium", "hard", "custom"}
envs: dict = {}


def _get_env(task_id: str) -> AutoCleanEnv:
    if task_id not in envs:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' not initialised. POST /reset?task_id={task_id} first."
        )
    return envs[task_id]


def _sanitise(obj):
    """Recursively replace NaN/Inf — prevents HTTP 500 on JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return 0.0
    if hasattr(obj, "model_dump"):
        return _sanitise(obj.model_dump())
    if hasattr(obj, "dict"):
        return _sanitise(obj.dict())
    return obj


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Health check — must return 200 for validator ping."""
    return {"message": "AutoClean-Pro OpenEnv is Live."}


@app.post("/reset")
async def reset(
    task_id:          str = "easy",
    bayesian_mode:    str = "auto",
    scarce_threshold: int = 50,
):
    if task_id not in VALID_TASKS:
        raise HTTPException(status_code=404, detail=f"Unknown task '{task_id}'.")
    envs[task_id] = AutoCleanEnv(
        task_id=task_id,
        bayesian_mode=bayesian_mode,
        scarce_threshold=scarce_threshold,
    )
    obs = envs[task_id].reset()
    return {"observation": _sanitise(obs), "info": {}}


@app.post("/step")
async def step(action: Action, task_id: str = "easy"):
    env    = _get_env(task_id)
    result = env.step(action)
    return _sanitise(result)


@app.get("/grader")
async def get_grader(task_id: str = "easy"):
    env   = _get_env(task_id)
    raw   = env.grader(silent=True)
    score = max(0.001, min(0.999, float(raw)))
    return _sanitise({
        "task_id":        task_id,
        "score":          score,
        "success":        (score > 0.99) if task_id == "hard" else (score > 0.98),
        "weighting_mode": env.state.weighting_mode,
        "dataset_regime": env.state.dataset_regime,
    })


@app.get("/state")
async def get_state(task_id: str = "easy"):
    """OpenEnv-compatible state endpoint — returns models.State Pydantic model."""
    env   = _get_env(task_id)
    state = env.get_state_model()
    return _sanitise({
        **state.model_dump(),
        "missing_total":  int(env.df.isnull().sum().sum()) if env.df is not None else 0,
        "total_rows":     len(env.df) if env.df is not None else 0,
        "available_tools": ToolRegistry.available(),
    })


@app.post("/upload")
async def upload_and_reset(
    file:             UploadFile = File(...),
    bayesian_mode:    str = "auto",
    scarce_threshold: int = 50,
):
    """
    Upload any messy CSV and run the cleaning agent against it.
    This makes AutoClean-Pro genuinely general-purpose — not just the
    3 bundled tasks. The agent will discover column names, dtypes, and
    missingness from the uploaded file at runtime.

    The uploaded CSV is treated as task_id="custom". The grader scores
    by checking whether all NaN values have been filled (no ground-truth
    required). flag_human is triggered if any column has >=35% missing.
    """
    contents = await file.read()
    try:
        import pandas as pd
        df_dirty = pd.read_csv(
            io.StringIO(contents.decode("utf-8")),
            na_values=["", " ", "nan", "NaN", "None", "null", "nan.0", "Nan"],
            keep_default_na=True,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    # Save to a temp file so AutoCleanEnv can read it
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, dir="/tmp"
    ) as tmp:
        df_dirty.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    # Patch paths and reset with a custom task
    env = AutoCleanEnv.__new__(AutoCleanEnv)
    env.task_id          = "custom"
    env.step_limit       = 15
    env._bayesian_mode   = bayesian_mode
    env._scarce_threshold = scarce_threshold
    env._paths           = {"custom": {"dirty": tmp_path, "clean": tmp_path}}
    env.df               = None
    env.target_df        = None
    env.weights          = None
    from environment import EpisodeState
    import uuid
    env._state = EpisodeState(task_id="custom")

    obs = env.reset()
    envs["custom"] = env

    return {
        "observation": _sanitise(obs),
        "info": {
            "task_id":         "custom",
            "rows":            len(env.df) if env.df is not None else 0,
            "columns":         list(env.df.columns) if env.df is not None else [],
            "missing_total":   int(env.df.isnull().sum().sum()) if env.df is not None else 0,
            "weighting_mode":  env.state.weighting_mode,
            "dataset_regime":  env.state.dataset_regime,
            "note": (
                "Use POST /step?task_id=custom to clean this dataset. "
                "GET /grader?task_id=custom for score."
            ),
        },
    }


@app.get("/tools")
async def list_tools():
    """Live tool registry — agents discover available tools at runtime."""
    return {"tools": ToolRegistry.available()}


@app.get("/download")
async def download_clean_csv(task_id: str = "easy"):
    """
    Download the current (cleaned) state of the dataset as a CSV file.
    Works for all task_ids including 'custom' uploaded CSVs.
    Call after /step to download the agent-cleaned version.
    """
    env = _get_env(task_id)
    if env.df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded. POST /reset first.")

    csv_bytes = env.df.to_csv(index=False).encode("utf-8")
    filename  = f"autoclean_{task_id}_cleaned.csv"

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/agent")
async def run_agent(task_id: str = "easy"):
    """
    Run the LLM agent IN-PROCESS against any active task — including
    custom uploaded CSVs.

    This is self-contained: the agent runs inside THIS request on THIS
    replica, using the same env object that /upload or /reset created.
    No subprocess, no HTTP hops, no cross-replica session loss.

    Workflow:
      1. POST /upload           → uploads CSV, initialises env
      2. POST /agent?task_id=custom  → agent cleans autonomously
      3. GET  /grader?task_id=custom → see the score
      4. GET  /download?task_id=custom → download cleaned CSV
    """
    import importlib
    import inference as inf_module
    importlib.reload(inf_module)   # pick up any env changes

    env = _get_env(task_id)        # raises 404 if not initialised

    obs        = _sanitise(env.reset())
    rewards    = []
    steps      = 0
    flag_done  = False
    log_lines  = []

    obs_dict = obs if isinstance(obs, dict) else (
        obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)
    )

    for step_idx in range(1, env.step_limit + 1):
        steps = step_idx
        action_dict = inf_module.get_agent_action(env, obs_dict, flag_done)

        if action_dict.get("tool") == "flag_human":
            flag_done = True

        result   = _sanitise(env.step(Action(**action_dict)))
        reward   = float(result.get("reward", 0.0))
        done     = bool(result.get("done", False))
        obs_dict = result.get("observation", {})
        if hasattr(obs_dict, "model_dump"):
            obs_dict = obs_dict.model_dump()
        err      = result.get("info", {}).get("last_action_error")

        line = (f"[STEP] step={step_idx} "
                f"action={__import__('json').dumps(action_dict, separators=(',',':'))} "
                f"reward={reward:.2f} done={str(done).lower()} "
                f"error={err or 'null'}")
        log_lines.append(line)
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
        rewards.append(reward)

        if done:
            break

    raw_score   = env.grader(silent=True)
    score       = max(0.001, min(0.999, float(raw_score)))
    success     = score > 0.99 if task_id == "hard" else (
                  score > 0.90 if task_id == "custom" else score > 0.98)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    end_line = (f"[END] success={str(success).lower()} steps={steps} "
                f"score={score:.2f} rewards={rewards_str}")
    log_lines.append(end_line)
    sys.stderr.write(end_line + "\n")
    sys.stderr.flush()

    return _sanitise({
        "status":  "Agent completed.",
        "success": success,
        "steps":   steps,
        "score":   score,
        "rewards": rewards,
        "log":     log_lines,
        "next_steps": {
            "score":    f"GET /grader?task_id={task_id}",
            "download": f"GET /download?task_id={task_id}",
        },
    })


@app.post("/baseline")
async def trigger_baseline(background_tasks: BackgroundTasks):
    """Run inference.py in background. Output appears in HF Space Logs tab."""
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
        "status":      "Inference started in background.",
        "instruction": "Monitor the HF Space Logs tab for [START]/[STEP]/[END] output.",
    }


@app.get("/baseline")
async def get_baseline():
    return {
        "message": (
            "Trigger POST /baseline to run the live inference agent. "
            "Results appear in the HF Space Logs tab."
        )
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()