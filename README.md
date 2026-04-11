---
title: AutoClean-Pro
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
short_description: Autonomous data cleaning RL env with HITL governance
---

# AutoClean-Pro: Adaptive Data Governance for RL Agents

**[Try the live API](https://himanshirawat0892-autoclean-pro.hf.space/docs)**

Data cleaning consumes 80% of ML engineer time. AutoClean-Pro is an OpenEnv RL environment where LLM agents learn to autonomously clean messy tabular datasets — deciding when to impute missing values statistically and when to escalate to a human reviewer under strict governance rules.

The key innovation is **Adaptive Bayesian Weighting**: missingness scores are amplified in scarce datasets (≤50 rows) where a single missing value carries disproportionate statistical risk — modelling real-world scenarios like rare disease registries, clinical trials, and sensor failure logs.

---

## How an Episode Works

1. `POST /reset?task_id=easy` — loads a dirty CSV and returns an observation
2. The agent reads `missing_report` (Bayesian-weighted missingness per column) and `schema_info`
3. `POST /step` — agent applies a cleaning tool to one column
4. Each step returns reward, updated observation, and done flag
5. `GET /grader` — deterministic score against ground-truth clean data

### Example: Easy Task

```
Observation after reset:
  missing_report: {"Age": 0.34, "Clicks": 0.14}
  weighting_mode: "bayesian_scarce"
  dataset_regime: "scarce_10rows"

Step 1: knn_impute(Age)       → reward: +1.50  (score=0.34, numeric, 0.15-0.34 band)
Step 2: median_impute(Clicks) → reward: +2.60  done=true
[END] success=true steps=2 score=1.00 rewards=1.50,2.60
```

### Example: Hard Task (Governance + Partial Imputation)

```
Observation after reset:
  missing_report: {"Survey_Response": 0.40, "Income_k": 0.28}
  weighting_mode: "bayesian_scarce"

Survey_Response: 40% missing → score ≥ 0.35 → flag_human required
Any imputation on ≥0.35 column → penalty: -5.0 reward

Step 1: flag_human(Survey_Response) → reward: +2.00
        Survey_Response score → 0.0 in next observation (column handled)
Step 2: knn_impute(Income_k)        → reward: +1.60
Step 3: finish                      → done=true
[END] success=true steps=3 score=1.00 rewards=2.00,1.60,0.00
Grader: 60% (governance flagged) + 40% (Income_k cleaned) = 1.0
```

---

## Adaptive Bayesian Weighting

The core technical innovation. Standard environments report flat missingness. AutoClean-Pro amplifies scores for scarce datasets with a critical constraint: **amplification never pushes a sub-threshold column into governance territory**.

```
amplification = 1 + (threshold - n) / (threshold × 2)
  n=10, threshold=50 → amp = 1.40

# Sub-threshold columns (flat < 0.35): amplify but cap at 0.34
weighted_pct = min(0.34, flat_pct × amp)

# Governance columns (flat ≥ 0.35): return flat_pct directly
weighted_pct = flat_pct
```

**Tool-selection bands:**

| Missing / 10 rows | Flat % | Weighted % | Tool |
|---|---|---|---|
| 1 / 10 | 0.10 | 0.14 | `median_impute` |
| 2 / 10 | 0.20 | 0.28 | `knn_impute` |
| 3 / 10 | 0.30 | 0.34 (capped) | `knn_impute` |
| 4 / 10 | 0.40 | 0.40 (governance) | `flag_human` |

**Three configurable modes:**

| Mode | Behaviour |
|---|---|
| `auto` | Detects dataset size, applies amplification for n ≤ 50 |
| `on` | Always amplifies — for clinical / rare-event datasets |
| `off` | Flat proportion — standard behaviour for large datasets |

---

## Agent: Chain-of-Thought with Guided Self-Consistency

The inference agent uses **CoT with guided self-consistency** rather than a single greedy call:

1. Sample 3 LLM responses (temperature=0.0 for first, 0.3 for diversity)
2. Each response produces `<think>...</think>` reasoning + JSON action
3. Select best action by: `urgency_score + 0.3 × consensus`
   - `urgency` = `missing_report[column]` from the environment
   - `consensus` = fraction of samples agreeing on (tool, column)

The system prompt uses an explicit **decision tree** with few-shot examples mapped to exact column names, eliminating ambiguity for borderline cases.

---

## Tool Registry

Tools self-register via decorator — the environment never hardcodes column names:

```python
@ToolRegistry.register("knn_impute")
def _knn_impute(df, col, params):
    k = int(params.get("n_neighbors", 5))
    imputed = KNNImputer(n_neighbors=k).fit_transform(...)
    return df, f"KNN impute (k={k}) on '{col}'."
```

New tools require only a decorated function — no changes to `AutoCleanEnv`. The agent discovers available tools from `GET /tools` at runtime.

---

## Tasks

| Task | Dataset | Key challenge | Correct episode | Success threshold |
|---|---|---|---|---|
| `easy` | 10 rows, Age (float) + Clicks (binary 0/1) | knn + median impute | `knn_impute(Age)` → `median_impute(Clicks)` | score > 0.98 |
| `medium` | 10 rows, Price (object) + Category (str) | Schema repair + imputation | `cast_type(Price)` → `knn_impute(Price)` → `mode_impute(Category)` | score > 0.98 |
| `hard` | 10 rows, 40% Survey_Response + 20% Income_k | Governance + partial clean | `flag_human(Survey_Response)` → `knn_impute(Income_k)` | score > 0.99 |

### Upload your own CSV

`POST /upload` accepts any messy CSV and runs the full Bayesian cleaning pipeline against it. Column names, dtypes, and missingness are discovered at runtime. Grader scores by NaN elimination — fraction of dirty cells filled — with no ground-truth file needed.

---

## Reward Shaping

```
Reward = ΔQuality + StrategyBonus + AccuracyBonus + RarityBonus − Penalties
```

| Component | Value |
|---|---|
| Cleaning gain (NaNs removed / total NaNs in column) | 0.0 – 1.0 |
| Strategy alignment (right tool for the score band) | +0.3 – +0.75 |
| Exact match with ground truth | +1.0 |
| `flag_human` on ≥35% column | +2.0 |
| Imputing a ≥35% column | −5.0 |
| Diverse tool usage | +0.1 |
| Back-to-back same tool | −2.0 |

Scores clamped to `(0.001, 0.999)` — strictly open interval as required by the validator.

---

## Architecture

![Architecture diagram](autoclean_system_architecture.svg)

### BaseEnv Pattern

```python
class BaseCleanEnv(ABC):
    @abstractmethod
    def reset(self) -> Observation: ...
    @abstractmethod
    def step(self, action: Action) -> Dict: ...
    @abstractmethod
    def grader(self, silent: bool = False) -> float: ...
    @property
    @abstractmethod
    def state(self) -> EpisodeState: ...
```

`AutoCleanEnv` inherits `BaseCleanEnv`. Column names are discovered from the loaded CSV at `reset()` — nothing hardcoded anywhere.

### EpisodeState

OpenEnv-compatible state with `episode_id` + `step_count`, plus Bayesian context:

```json
{
  "episode_id": "uuid",
  "step_count": 3,
  "weighting_mode": "bayesian_scarce",
  "dataset_regime": "scarce_10rows",
  "history": ["flag_human", "knn_impute", "finish"],
  "flagged_cols": ["Survey_Response"]
}
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check — returns 200 for validator |
| `/reset` | POST | Start episode. Params: `task_id`, `bayesian_mode`, `scarce_threshold` |
| `/step` | POST | Execute action: `{"tool": "...", "column": "...", "params": {}}` |
| `/grader` | GET | Deterministic score + weighting context |
| `/state` | GET | Full `EpisodeState` — OpenEnv compatible |
| `/tools` | GET | Live tool registry — agent discovers tools at runtime |
| `/upload` | POST | Upload any CSV — Bayesian cleaning without ground truth |
| `/baseline` | POST | Run `inference.py` in background; output in HF Space Logs tab |
| `/docs` | GET | Interactive Swagger UI |

---

## Setup and Running

```bash
# Install dependencies
uv sync

# Start the API server (port 7860)
uv run uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run the inference agent
uv run python inference.py
```

### Expected inference output

```
[START] task=easy env=autoclean_benchmark model=Qwen/Qwen2.5-7B-Instruct
[STEP] step=1 action={"tool":"knn_impute","column":"Age","params":{}} reward=1.50 done=false error=null
[STEP] step=2 action={"tool":"median_impute","column":"Clicks","params":{}} reward=2.60 done=true error=null
[END] success=true steps=2 score=1.00 rewards=1.50,2.60

[START] task=medium env=autoclean_benchmark model=Qwen/Qwen2.5-7B-Instruct
[STEP] step=1 action={"tool":"cast_type","column":"Price","params":{"target_dtype":"float64"}} reward=0.10 done=false error=null
[STEP] step=2 action={"tool":"knn_impute","column":"Price","params":{}} reward=2.60 done=false error=null
[STEP] step=3 action={"tool":"mode_impute","column":"Category","params":{}} reward=2.85 done=true error=null
[END] success=true steps=3 score=1.00 rewards=0.10,2.60,2.85

[START] task=hard env=autoclean_benchmark model=Qwen/Qwen2.5-7B-Instruct
[STEP] step=1 action={"tool":"flag_human","column":"Survey_Response","params":{}} reward=2.00 done=false error=null
[STEP] step=2 action={"tool":"knn_impute","column":"Income_k","params":{}} reward=1.60 done=false error=null
[STEP] step=3 action={"tool":"finish","column":null,"params":{}} reward=0.00 done=true error=null
[END] success=true steps=3 score=1.00 rewards=2.00,1.60,0.00
```

---


---

## How to Use — Manual Cleaning Workflow

You can clean any messy CSV through the live API without writing code. Use the Swagger UI at [`/docs`](https://himanshirawat0892-autoclean-pro.hf.space/docs) or the curl commands below.

### Step 1 — Upload your CSV

```bash
curl -X POST "https://himanshirawat0892-autoclean-pro.hf.space/upload" \
  -F "file=@your_data.csv;type=text/csv"
```

The response tells you which columns are dirty and what tools to use:

```json
{
  "observation": {
    "missing_report": {"Salary": 0.183},
    "schema_info":    {"Salary": "float64"},
    "weighting_mode": "standard_uniform",
    "dataset_regime": "standard_322rows"
  },
  "info": {
    "task_id": "custom",
    "missing_total": 59,
    "note": "Use POST /step?task_id=custom to clean this dataset."
  }
}
```

### Step 2 — Read the missing report and pick a tool

| Score | Band | Tool to use |
|---|---|---|
| ≥ 0.35 | Governance | `flag_human` — too risky to impute |
| 0.15 – 0.34 | Significant | `knn_impute` |
| 0.00 – 0.14 | Small gap | `median_impute` (numeric) or `mode_impute` (categorical) |
| dtype=object, numeric name | Schema mismatch | `cast_type` first, then impute |

### Step 3 — Apply cleaning steps

```bash
# Impute Salary (score=0.183 → knn_impute)
curl -X POST "https://himanshirawat0892-autoclean-pro.hf.space/step?task_id=custom" \
  -H "Content-Type: application/json" \
  -d '{"tool":"knn_impute","column":"Salary","params":{}}'
```

Repeat for each dirty column until all `missing_report` scores are 0.0.

### Step 4 — Check your score

```bash
curl "https://himanshirawat0892-autoclean-pro.hf.space/grader?task_id=custom"
```

Returns a score between 0 and 1 based on the fraction of dirty cells now filled.

### Step 5 — Download the cleaned CSV

```bash
curl "https://himanshirawat0892-autoclean-pro.hf.space/download?task_id=custom" \
  -o my_data_cleaned.csv
```

The file contains your original data with all imputed values filled in.

---

### Full example: Hitters baseball salary dataset

```bash
# 1. Upload
curl -X POST "https://himanshirawat0892-autoclean-pro.hf.space/upload" \
  -F "file=@Hitters.csv;type=text/csv"
# → Salary: 59/322 missing (18.3%) → score=0.183 → knn_impute

# 2. Impute Salary
curl -X POST "https://himanshirawat0892-autoclean-pro.hf.space/step?task_id=custom" \
  -H "Content-Type: application/json" \
  -d '{"tool":"knn_impute","column":"Salary","params":{}}'

# 3. Score
curl "https://himanshirawat0892-autoclean-pro.hf.space/grader?task_id=custom"
# → {"score": 0.999, "success": true}

# 4. Download
curl "https://himanshirawat0892-autoclean-pro.hf.space/download?task_id=custom" \
  -o hitters_cleaned.csv
```

> **Note on imputation quality:** The agent uses single-column KNN imputation.
> For small datasets (≤50 rows) this works well because Bayesian weighting
> amplifies the signal from each missing cell. For large datasets like Hitters
> (322 rows), single-column KNN produces a mean-equivalent fill — all missing
> cells receive the same value (the mean of non-null values). This is statistically
> valid but not as precise as multi-feature KNN. A future upgrade will use
> all available numeric columns as KNN features for uploaded datasets.

## Project Structure

```
├── data/                  # Dirty and clean CSV pairs (6 files)
├── server/
│   └── app.py             # FastAPI server (port 7860)
├── environment.py         # BaseCleanEnv + AutoCleanEnv + ToolRegistry
├── logic.py               # Adaptive Bayesian weighting + reward functions
├── models.py              # Pydantic V2 schemas (Action, Observation, State)
├── inference.py           # CoT + guided self-consistency agent
├── generate_data.py       # Synthetic dataset generator
├── pyproject.toml         # Dependencies (uv)
├── openenv.yaml           # OpenEnv spec
└── .gitattributes         # Line ending normalization (LF)
```