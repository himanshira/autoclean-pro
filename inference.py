"""
AutoClean-Pro — inference script v2

Agent strategy: Chain-of-Thought with Guided Self-Consistency
--------------------------------------------------------------
For each step, we sample N=3 LLM responses (n=1 for speed on simple cases).
Each response produces a <think>...</think> block + JSON action.
The winning action is chosen by the highest missing_report score for the
targeted column — the environment provides the tie-breaker, not majority vote.

This is "guided self-consistency":
  - Pure self-consistency: take the majority vote of N responses
  - Guided self-consistency: take the response that targets the column
    with the highest environmental urgency score

Why this is better for data cleaning:
  - The LLM sometimes hallucinates a clean column when a dirty one exists
  - 3 samples make that error much less likely
  - The missing_report score is a grounded, deterministic signal
  - No extra grader calls — selection uses the current observation only
"""

import os
import json
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from environment import AutoCleanEnv, ToolRegistry
from models import Action
from logic import validate_cleaning_strategy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-7B-Instruct")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
# HF_TOKEN is the primary hackathon variable; API_KEY is the Phase 2 proxy variable
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

if API_KEY is None:
    raise ValueError(
        "API_KEY or HF_TOKEN environment variable is required. "
        "Set with: export API_KEY=... or export HF_TOKEN=hf_..."
    )

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

VALID_TOOLS = set(ToolRegistry.available())

# Self-consistency samples — 3 gives meaningful diversity without latency cost
SC_SAMPLES = 3

# ---------------------------------------------------------------------------
# Logging — stdout ONLY: [START] / [STEP] / [END]
# ---------------------------------------------------------------------------

def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(step: int, action_str: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    error_val = (
        error if (error and str(error).lower() not in ("none", "null", ""))
        else "null"
    )
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float],
            score: float = 0.0) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else ""
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _warn(msg: str) -> None:
    sys.stderr.write(f"[WARN] {msg}\n")
    sys.stderr.flush()


def _think(reasoning: str) -> None:
    """CoT reasoning goes to stderr only — stdout is strictly [START]/[STEP]/[END]."""
    sys.stderr.write(f"[THINK] {reasoning[:300]}\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Observation → plain dict
# ---------------------------------------------------------------------------

def obs_to_dict(obs: Any) -> Dict[str, Any]:
    if isinstance(obs, dict):
        return obs
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if hasattr(obs, "dict"):
        return obs.dict()
    return {}


# ---------------------------------------------------------------------------
# Success evaluation
# ---------------------------------------------------------------------------

def evaluate_success(env: AutoCleanEnv) -> Tuple[bool, float]:
    try:
        score = env.grader(silent=True)
        if env.task_id == "hard":
            # 60% flag_human + 40% income clean = 1.0 → clamped 0.999
            return bool(score > 0.99), float(score)
        if env.task_id == "custom":
            # NaN-elimination grader: success if >90% of dirty cells filled
            return bool(score > 0.90), float(score)
        # easy / medium: np.isclose vs ground truth
        return bool(score > 0.98), float(score)
    except Exception as e:
        _warn(f"evaluate_success failed: {e}")
        return False, 0.0


# ---------------------------------------------------------------------------
# LLM proxy warm-up — Phase 2 requirement
# ---------------------------------------------------------------------------

def warmup_llm_proxy() -> None:
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            temperature=0.0,
            timeout=10,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# System prompt — dynamically built from observation context
# ---------------------------------------------------------------------------

def build_system_prompt(obs: Dict[str, Any]) -> str:
    mode    = obs.get("weighting_mode", "standard_uniform")
    regime  = obs.get("dataset_regime", "unknown")
    schema  = obs.get("schema_info", {})
    missing = obs.get("missing_report", {})
    tools   = obs.get("available_tools", sorted(VALID_TOOLS))

    col_lines = []
    for col, dtype in schema.items():
        score = missing.get(col, 0.0)
        if score > 0:
            col_lines.append(
                f"  - {col} (dtype={dtype}, weighted_missing={score:.3f})"
            )
    cols_ctx = "\n".join(col_lines) if col_lines else "  all columns are clean"

    if "bayesian" in mode:
        mode_guidance = (
            "BAYESIAN SCARCE MODE is active. Scores are amplified but CAPPED at 0.34\n"
            "for columns with actual missingness < 35%. IMPORTANT boundaries:\n"
            "  score 0.00–0.14  → median_impute\n"
            "  score 0.15–0.34  → knn_impute  (score 0.34 means knn, NOT flag_human)\n"
            "  score exactly >= 0.35 → flag_human ONLY\n"
            "CRITICAL: A score of 0.34 means knn_impute — NOT flag_human.\n"
            "flag_human is ONLY for scores >= 0.35. Never flag columns with score <= 0.34."
        )
    else:
        mode_guidance = (
            "STANDARD MODE is active. Scores are flat proportions:\n"
            "  score 0.00–0.14  → median_impute\n"
            "  score 0.15–0.34  → knn_impute\n"
            "  score >= 0.35    → flag_human ONLY"
        )

    return f"""You are a data-cleaning agent for a {regime} dataset.

{mode_guidance}

Available tools: {tools}

Columns needing attention (from live observation):
{cols_ctx}

INSTRUCTIONS:
1. Think step-by-step inside <think>...</think> tags. Be explicit about:
   - Which column has the highest urgency (highest weighted_missing score)
   - Why you chose this specific tool for this column and this mode
   - Whether the dtype is categorical (object) or numeric
2. After </think>, output exactly ONE JSON action on a new line.

DECISION TREE — follow exactly, stop at first match for TARGET column:

STEP 1: If ALL missing_report scores are 0.0 → finish
  {{"tool":"finish","column":null,"params":{{}}}}

STEP 2: Pick TARGET = column with HIGHEST weighted_missing score.

STEP 3: Choose tool for TARGET:

  [A] TARGET score ≥ 0.35 → flag_human, regardless of dtype
      {{"tool":"flag_human","column":"TARGET","params":{{}}}}
      After flagging, that column score becomes 0. Next step: go to STEP 1.
      *** dtype=object does NOT change this. Survey_Response=0.40 → flag_human ***

  [B] TARGET dtype is object/str:
      → Name sounds numeric? (price/cost/income/salary/amount/rate/value/revenue)
        YES → cast_type {{"tool":"cast_type","column":"TARGET","params":{{"target_dtype":"float64"}}}}
        NO  → mode_impute {{"tool":"mode_impute","column":"TARGET","params":{{}}}}
      *** knn_impute is NEVER valid for object/str columns ***

  [C] TARGET dtype is numeric (float64/int64):
      → score 0.15–0.34 → knn_impute {{"tool":"knn_impute","column":"TARGET","params":{{}}}}
      → score 0.00–0.14 → check if all non-null values are 0 or 1:
          YES (binary) → mode_impute  {{"tool":"mode_impute","column":"TARGET","params":{{}}}}
          NO (continuous) → median_impute {{"tool":"median_impute","column":"TARGET","params":{{}}}}

FEW-SHOT EXAMPLES (these are the exact columns you will see — use these patterns):
  Survey_Response score=0.40 dtype=str   → [A] flag_human     (≥0.35, always flag)
  Income_k        score=0.28 dtype=float → [C] knn_impute     (numeric, 0.15-0.34)
  Age             score=0.34 dtype=float → [C] knn_impute     (numeric, 0.15-0.34)
  Price           score=0.28 dtype=str   → [B] cast_type      (numeric-named object)
  Category        score=0.14 dtype=str   → [B] mode_impute    (categorical-named object)
  Clicks          score=0.14 dtype=float → [C] median_impute  (numeric, 0.00-0.14)

OUTPUT: one JSON on a new line, nothing else after </think>:
{{"tool": "<tool>", "column": "<col>", "params": {{}}}}"""


# ---------------------------------------------------------------------------
# CoT response parser
# ---------------------------------------------------------------------------

def parse_cot_response(raw: str) -> Tuple[Optional[Dict], str]:
    """
    Robust parser for Qwen2.5 CoT output.
    Handles: markdown fences, unclosed </think>, Qwen special tokens,
    JSON after prose, bare JSON, nested params braces.
    """
    # Strip Qwen chat special tokens
    raw = re.sub(r"<\|im_start\|>\w*\n?", "", raw)
    raw = re.sub(r"<\|im_end\|>", "", raw)
    raw = raw.strip()

    # Extract think block — closing tag is OPTIONAL (Qwen often omits it)
    thinking = ""
    think_open = raw.find("<think>")
    if think_open != -1:
        think_content_start = think_open + len("<think>")
        think_close = raw.find("</think>", think_content_start)
        if think_close != -1:
            thinking = raw[think_content_start:think_close].strip()
            raw = raw[think_close + len("</think>"):].strip()
        else:
            # No closing tag: grab everything before first { as thinking
            first_brace = raw.find("{", think_content_start)
            if first_brace != -1:
                thinking = raw[think_content_start:first_brace].strip()
                raw = raw[first_brace:]
            else:
                thinking = raw[think_content_start:].strip()
                raw = ""

    # Strip markdown code fences
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = raw.replace("```", "").strip()

    # Find outermost JSON object via brace depth counting
    start = raw.find("{")
    if start == -1:
        return None, thinking

    depth, end = 0, -1
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        return None, thinking

    json_str = raw[start:end + 1]
    try:
        return json.loads(json_str), thinking
    except json.JSONDecodeError:
        json_str = json_str.replace("'", '"')
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
        try:
            return json.loads(json_str), thinking
        except Exception:
            return None, thinking




# ---------------------------------------------------------------------------
# Guided self-consistency selection
# ---------------------------------------------------------------------------

def select_best_action(
    candidates: List[Dict[str, Any]],
    missing_report: Dict[str, float],
    schema: Dict[str, str],
) -> Dict[str, Any]:
    """
    Select the best action from N candidates using guided self-consistency.

    Selection criteria (in order):
    1. Discard invalid/unknown tools
    2. Among valid candidates, score each by:
         urgency   = missing_report[column] (higher = more urgent)
         consensus = how many candidates agree on this (tool, column) pair
    3. Pick the candidate with highest urgency × consensus weight

    This combines environmental grounding (urgency) with LLM agreement
    (consensus) to get the best of both self-consistency variants.
    """
    valid = []
    for c in candidates:
        tool = str(c.get("tool", ""))
        col  = c.get("column")
        if tool not in VALID_TOOLS:
            continue
        # Safety: block knn_impute on object columns
        if tool == "knn_impute" and col and schema.get(col, "") == "object":
            c = {"tool": "mode_impute", "column": col, "params": {}}
        valid.append(c)

    if not valid:
        return {"tool": "finish", "column": None, "params": {}}

    # Score each candidate
    # consensus_weight: how many of the N samples agree on (tool, column)
    pair_counts: Counter = Counter()
    for c in valid:
        pair_counts[(c.get("tool"), c.get("column"))] += 1

    def score(c: Dict) -> float:
        col      = c.get("column")
        urgency  = missing_report.get(col, 0.0) if col else 0.0
        consensus = pair_counts[(c.get("tool"), col)] / SC_SAMPLES
        # finish has no urgency — only select it if truly nothing is dirty
        if c.get("tool") == "finish":
            urgency = 0.0
        return urgency + 0.3 * consensus   # urgency dominates, consensus breaks ties

    return max(valid, key=score)


# ---------------------------------------------------------------------------
# Main agent action — CoT + guided self-consistency
# ---------------------------------------------------------------------------

def get_agent_action(
    obs: Dict[str, Any],
    env: AutoCleanEnv,
    flag_human_done: bool,
) -> Dict[str, Any]:
    """
    Sample SC_SAMPLES LLM responses with CoT, then select the best
    action via guided self-consistency against the missing_report scores.

    flag_human_done only prevents a second flag_human call — it does NOT
    force finish. The agent continues cleaning other dirty columns after
    flagging a governance column.
    """
    missing_report = obs.get("missing_report", {})
    schema         = obs.get("schema_info", {})

    # Early exit: nothing to clean
    if not any(v > 0 for v in missing_report.values()):
        return {"tool": "finish", "column": None, "params": {}}

    system_prompt = build_system_prompt(obs)
    user_msg = (
        f"Current observation:\n{json.dumps(obs, indent=2)}\n\n"
        "Think step by step about which column needs the most urgent attention, "
        "then output your JSON action."
    )

    candidates: List[Dict] = []
    all_thinking: List[str] = []

    # Sample N responses
    for i in range(SC_SAMPLES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=400,
                temperature=0.3 if i > 0 else 0.0,  # first sample greedy, rest diverse
                timeout=30,
            )
            raw     = response.choices[0].message.content
            action, thinking = parse_cot_response(raw)
            if thinking:
                all_thinking.append(f"[Sample {i+1}] {thinking}")
            if action:
                candidates.append(action)
        except Exception as exc:
            _warn(f"LLM sample {i+1} failed: {exc}")

    # Log all thinking to stderr
    if all_thinking:
        _think(" | ".join(all_thinking))

    # Fall back to deterministic if all LLM calls failed
    if not candidates:
        _warn("All LLM samples failed — using deterministic fallback")
        return _deterministic_fallback(obs, schema, missing_report, flag_human_done)

    return select_best_action(candidates, missing_report, schema)


def _deterministic_fallback(
    obs: Dict[str, Any],
    schema: Dict[str, str],
    missing_report: Dict[str, float],
    flag_human_done: bool = False,
) -> Dict[str, Any]:
    """
    Rule-based fallback using only observation data — no hardcoded columns.
    """
    # Governance first — but only if not already flagged
    if not flag_human_done:
        for col, score in missing_report.items():
            if score >= 0.35:
                return {"tool": "flag_human", "column": col, "params": {}}

    # Cast numeric-looking object columns
    numeric_patterns = {
        "price", "income", "age", "salary", "score",
        "count", "amount", "rate", "value", "clicks",
    }
    for col, dtype in schema.items():
        if dtype == "object" and missing_report.get(col, 0) > 0:
            if any(p in col.lower() for p in numeric_patterns):
                return {"tool": "cast_type", "column": col,
                        "params": {"target_dtype": "float64"}}

    # Mode impute remaining object columns
    for col, score in missing_report.items():
        if score > 0 and schema.get(col, "") == "object":
            return {"tool": "mode_impute", "column": col, "params": {}}

    # Median/KNN for numeric columns
    numeric_dirty = {
        col: score for col, score in missing_report.items()
        if score > 0 and schema.get(col, "") != "object"
    }
    if numeric_dirty:
        col  = max(numeric_dirty, key=lambda c: numeric_dirty[c])
        tool = "knn_impute" if numeric_dirty[col] >= 0.15 else "median_impute"
        return {"tool": tool, "column": col, "params": {}}

    return {"tool": "finish", "column": None, "params": {}}


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task_id: str) -> None:
    rewards:         List[float] = []
    steps            = 0
    flag_human_done  = False
    env: AutoCleanEnv = None

    log_start(task_id, "autoclean_benchmark", MODEL_NAME)

    try:
        env = AutoCleanEnv(
            task_id=task_id,
            bayesian_mode="auto",
            scarce_threshold=50,
        )
        obs = obs_to_dict(env.reset())

        for step_idx in range(1, env.step_limit + 1):
            steps = step_idx

            action_dict = get_agent_action(obs, env, flag_human_done)

            if action_dict.get("tool") == "flag_human":
                flag_human_done = True
            # Note: flag_human_done does NOT exit the loop.
            # Agent continues to clean remaining dirty columns.

            result = env.step(Action(**action_dict))

            reward = float(result.get("reward", 0.0))
            done   = bool(result.get("done", False))
            obs    = obs_to_dict(result.get("observation", {}))
            info   = result.get("info", {})
            err    = info.get("last_action_error")

            log_step(
                step_idx,
                json.dumps(action_dict, separators=(",", ":")),
                reward, done, err,
            )
            rewards.append(reward)

            if done:
                break

    except Exception as exc:
        sys.stderr.write(f"[ERROR] run_task({task_id}): {exc}\n")
        sys.stderr.flush()

    finally:
        success     = False
        final_score = 0.0
        if env is not None:
            success, final_score = evaluate_success(env)
            env.close()
        log_end(success, steps, rewards, final_score)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    warmup_llm_proxy()   # Phase 2: register proxy usage

    if len(sys.argv) > 1:
        run_task(sys.argv[1])
    else:
        for task in ["easy", "medium", "hard"]:
            run_task(task)