---
title: AutoClean-Pro
emoji: 🧹
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---
# **AutoClean-Pro: Hardware-Aware Data Governance**

## **Motivation**
In real-world AI pipelines, data cleaning consumes **80% of engineer time**. The RL environments currently are more focused on toy games; **AutoClean-Pro** evaluates if an AI agent can autonomously transform "raw, messy" data into "ML-ready" features while adhering to strict **Data Governance** standards.

Our environment tests the "Decision Intelligence" of an agent knowing when to automate (impute) versus when to escalate to a human (flag), simulating a high-stakes AI infrastructure where incorrect data entry can lead to hardware-level failures.

## **Action & Observation Spaces**
### **Action Space (Discrete)**
The agent interacts with the data via a suite of specialized tools:
 
 1. ```knn_impute```: k-Nearest Neighbours (iterative, hardware-intensive).
 2. ```mode_impute / median_impute```: Statistical central tendency imputation.
 3. ```cast_type```: Schema alignment (e.g., String to Float).
 6. ```flag_human```: The Governance Tool. Required when data integrity is compromised.
 7. ```finish```: Submits the final dataset for grading.

### **Observation Space (JSON)**
The agent receives a rich state representation:
 1. ```data_preview```: A 10-row window of the current dataframe (List of Records).
 2. ```missing_report```: A **Bayesian-weighted dictionary** of null importance per column.
 3. ```schema_info```:Current data types (e.g., float64, object) to prevent type-mismatch errors.
 4. ```message```: Direct Feedback from the environment (e.g., "Governance Blocked or "Type Cast Successful").

## **Technical Innovation: Bayesian Missingness & Governance**
- **Bayesian-Weighted Reporting**
We apply a Bayesian weight to null counts to prevent "Small-Dataset Noise." In micro-datasets, a single missing value can disproportionately skew the "Data Health Score." Our weighted approach ensures the agent isn't "over-reacting" to isolated gaps in smaller columns.

- **The 40% Governance Rule (HITL)**
Located in ```logic.py```, this acts as a hard gatekeeper.
- **Policy**: If a column has >= **40% missing data**, any attempt to impute is penalized (**-2.0 reward**).
- **Requirement**: The agent must use ```flag_human``` to move the data to a "Review Required" state, mirroring real-world compliance where statistical guessing high volumes is prohibited.  


## **Deterministic Grader Design**
Our environment utilizes a **Dual-Criteria Grader** to ensure scientific reproducibility:
 1. **Mathematical Fidelity**: Compares results against ground-truth clean data. For KNN calculations, we utilize string-matching on rounded values to ensure floating-point variance accross different hardware (CPU vs TPU) does not penalize the agent.
 2. **Governance Alignment (Hard Task)**: In the "Hard" phase, the grader specifically checks for the "Review Required" string in the ```Survey_Response``` column. Failure to flag results in a 0.0 score for that task.

## **Baseline Inference (inference.py)**
The baseline utilizes **Zero-Shot Chain-of-Thought (COT)**. The agent is strictly governed by a decision tree provided in the SYSTEM_PROMPT:
- **Weight >= 0.35**: Flag Human.
- **0.15 <= Weight < 0.35**: KNN Impute.
- **0.05 <= Weight < 0.15**: Median Impute.

The agent outputs its thinking process that why it chose a particular tool to perform a particular action to clean the dataset. Basis that the agent is penalized or rewarded. 

## **System Architecture & Middleware**
AutoClean-Pro is engineered for **High-Availability** and **Interoperability** with a Hardware-Aware philosophy, specifically optimized for high-throughput, low-latency execution on constrained **2 vCPU / 8GB RAM** environments.:
1. **CORS Middleware**: Enabled to allow cross-origin requests from remote AI agents and external monitoring dashboards.
2. **Request Logging**: Custom middleware tracks agent "Think Time" and ensures that the 15-step limit is strictly enforced at the API layer.
3. **Error Handling**: Global exception handlers prevent server crashes during "Multi-Mode" evaluation, ensuring the environment remains responsive even if an agent sends a malformed action.
4. **Green AI Efficiency**: By minimizing the computational footprint of the cleaning agents, we reduce the energy overhead per data repair task.
5. **Modular API Design**: Utilizing a FastAPI backend to ensure that environment resets and step executions are decoupled from the heavy LLM inference.
![System Architecture Diagram](./Architecture%20Diagram)


## **Reward Shaping: Dense Signal and Policy Alignment**
We utilize a non-binary reward function to provide a dense signal throughout the trajectory, guiding the agent toward the 100% accuracy target:
     $$Reward = \Delta Quality + RarityBonus - RepetitionPenalty$$
- ```Cleaning Gain (\Delta Quality)```: Positive reward proportional to the percentage of NaNs removed.
- ```The 40% Governance Rule```: The agent is only rewarded (+2.0) for using flag_human.
-  ```Rarity Bonus```: A small incentive of $+0.1$ for utilizing diverse tools, preventing "tool-spamming".
-  ```Redundancy Penalty```: A negative reward (-0.1) if the agent tries to clean an already cleaned column or repeats an ineffective action.


## **Setup and Usage**
### **Local Installation**
    #Install dependencies using uv (recommended)
    uv sync
    uv lock

### **Running the Environment Server**
    #Start the OpenEnv-compliant FastAPI server
    python -m server.app.py

### **Running the Inference**
    python inference.py


## **Project Structure**
    ├── data/               # Dirty and Clean CSV pairs 
    ├── server/
    │   └── app.py          # FastAPI Entry Point (Port 7860)
    ├── environment.py      # Core RL Logic & Grader
    ├── models.py           # Pydantic V2 Schemas
    ├── inference.py         # Reproducible Inference Script
    ├── pyproject.toml      # Project Metadata & Entry Points
    └── openenv.yaml        # OpenEnv Specification File


