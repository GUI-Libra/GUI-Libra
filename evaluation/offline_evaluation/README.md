## Overview

This repository provides **offline evaluation pipelines for GUI agents** on two benchmarks:

- **Multimodal-Mind2Web-v2** 
- **AndroidControl-v2** 

Based on UGound, we further provide data process to add natural-language descriptions in MM-Mind2Web and filter noisy data in AndroidControl. You can also evaluate baselines such as GPT series and Qwen2.5VL/Qwen3VL/GUI-R1/Aguvis.

## Repository Structure

- **`Multimodal-Mind2Web/`**
  - `sample.py` – sample tasks from the original dataset
  - `make_blocks.py` – split full screenshots into `1280x1000` blocks
  - `gpt_plan.py` – generate plans with GPT (OpenAI API)
  - `guilibra_plan.py` / `glm_plan.py` / `aguvis_uitars_plan.py` / `uir1_plan.py` – generate plans with local or third-party models via OpenAI-compatible endpoints
  - `extract_grounding_query.py` – extract element-grounding queries from plans
  - `eval.py` / `eval_rl.py` / `eval_rl_bestofn.py` – evaluate Element Accuracy, Operation F1, Step Success Rate, and RL-style metrics
  - `gpt_plan_gen*.sh` – batch scripts to run planning for different models / splits
  - `inference.sh` / `extract_query.sh` / `eval_pipeline.sh` – example pipelines for grounding + evaluation
  - `README.md` – detailed instructions specific to this benchmark

- **`AndroidControl/`**
  - `sample.py` – sample and preprocess AndroidControl episodes
  - `gpt_plan.py` – GPT-based planning for Android (OpenAI API)
  - `guilibra_plan.py` – local JSON-style planner via OpenAI-compatible endpoint
  - `aguvis_uitars_plan.py` – local planners for Aguvis / UI-TARS families
  - `guir1_plan.py` – GUI-R1 / UI-R1 style planners
  - `extract_grounding_query.py` – extract grounding queries from plans
  - `eval.py` / `eval_best_of_n.py` / `eval_rl.py` – evaluate step accuracy, grounding accuracy, and best-of-n performance
  - `gpt_plan_gen*.sh` / `gpt_plan_gen_guir1.sh` – batch scripts to generate plans for different agents
  - `extract_query.sh` / `inference*.sh` / `get_score*.sh` – example grounding + evaluation pipelines
  - `README.md` – AndroidControl-specific usage

- **Root-level scripts**
  - `filter_eval_data.py` – filter AndroidControl evaluation data by:
    - **action_match** mode: check whether the action matches the instruction
    - **coord** mode: check whether predicted coordinates fall into a predicted bounding box
  - `gen_action_description_MMM2web.py` – generate natural language action descriptions for Multimodal-Mind2Web

## Dependencies & Environment
The base environment instralled in EasyR1 folder works for this environment.

### API keys & Endpoints

- **OpenAI API**:
  - Set `OPENAI_API_KEY` in your environment:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

- **Local / vLLM models**:
  - All local planners (`guilibra_plan.py`, `aguvis_uitars_plan.py`, `guir1_plan.py`) assume an **OpenAI-compatible HTTP endpoint**:
    -  vllm serve ${model_name} --port 8000 --dtype bfloat16 --max-model-len 8092  
    - `base_url="http://localhost:<port>/v1"`
    - `api_key` read from environment or set as needed for your deployment


## Datasets

### Download data from Huggingface



### Multimodal-Mind2Web

- Download the original dataset and precomputed blocks from the links in `Multimodal-Mind2Web/README.md`.
- Folder conventions:
  - Sample blocks JSONL: `data/samples/cross_{split}_blocks_natural.jsonl` (`split ∈ {domain, website, task}`)
  - Block images: `path/to/blocks_images/cross_{split}`

### AndroidControl

- Use the filtered JSON data and screenshot folders as described in `AndroidControl/README.md`:
  - Data JSON: e.g. `AndroidControl/data/500_steps_filtered.json`
  - Images: e.g. `path/to/AndroidControl_images`

## Quick Start

Quick Start uses **`plan_gen_guilibra.sh`** in each benchmark folder. It runs `guilibra_plan.py` against a local model served via an OpenAI-compatible API (e.g. vLLM). Ensure the model is already serving before running the script.

### Prerequisites

- **vLLM (or compatible) server** running your model, e.g.:
  ```bash
  vllm serve path/to/your/model --port 8000 --dtype bfloat16 --max-model-len 8092
  ```
- **Data in place**:
  - **Multimodal-Mind2Web**: `Multimodal-Mind2Web/data/samples/cross_{split}_blocks_natural.jsonl` and block images under `blocks_dir/cross_{split}`.
  - **AndroidControl**: `AndroidControl/data/500_steps_filtered.json` and screenshot images under `screenshot_dir`.

---

### 1. Multimodal-Mind2Web: `plan_gen_guilibra.sh`

**Location:** `Multimodal-Mind2Web/plan_gen_guilibra.sh`

**Parameters to set (top of script):**

| Variable | Meaning | Example |
|----------|--------|--------|
| `dir` | Output subfolder name under `data/` | `'2026_2_3'` |
| `add_template` | Add JSON template to prompt (0/1) | `1` |
| `reasoning` | Include reasoning in prompt (0/1) | `1` |
| `blocks_dir` | Root path of block images | `path/to/blocks_images` (e.g. from GUI-Libra/Offline-Evaluation) |
| `model_paths` | Model name/path (one per GPU) | `('path/to/your/model')` |
| `exp_names` | Experiment name for outputs | `('exp_name')` |
| `ports` | vLLM ports, one per entry in `model_paths` | `(8000 8001 ...)` |

**Input data:** Script uses `data/samples/cross_{split}_blocks_natural.jsonl` for `split ∈ {domain, website, task}`.

**Run:**

```bash
cd Multimodal-Mind2Web

# 1) Edit plan_gen_guilibra.sh: set blocks_dir, model_paths, exp_names, ports
# 2) Start vLLM (or compatible) on each port
# 3) Run:

bash plan_gen_guilibra.sh
```

Outputs: `data/${dir}/${exp_name}/cross_${split}/plan_${id}_temperature${temperature}.jsonl`.

**Evaluation (RL-style metrics):** Run `get_answer_rl.sh` to score plan files without a separate grounding step. Set `dir` to the path that contains per-split plan folders (e.g. `data/exp_name`), and `relative_coord` to `0` for Qwen2.5-VL–style absolute coordinates or `1` for Qwen3-VL–style relative coordinates. The script runs `eval_rl.py` for pass@1 (single plan per task) and `eval_rl_bestofn.py` for pass@4 over `cross_domain`, `cross_website`, and `cross_task`. Expected layout: `${dir}/cross_${key}/plan_0_temperature0.0.jsonl` and `plan_0_temperature1.0.jsonl`.

```bash
cd Multimodal-Mind2Web
# Edit get_answer_rl.sh: set dir= and relative_coord=
bash get_answer_rl.sh
```

---

### 2. AndroidControl: `plan_gen_guilibra.sh`

**Location:** `AndroidControl/plan_gen_guilibra.sh`

**Parameters to set (top of script):**

| Variable | Meaning | Example |
|----------|--------|--------|
| `dir` | Output subfolder name under `data/` | `'2026_2_3'` |
| `add_template` | Add JSON template to prompt (0/1) | `1` |
| `reasoning` | Include reasoning in prompt (0/1) | `1` |
| `screenshot_dir` | Root path of Android screenshots | `'path/to/AndroidControl_images'` |
| `model_paths` | Model name/path (one per GPU) | `('path/to/your/model')` |
| `exp_names` | Experiment name for outputs | `('my_experiment')` |
| `ports` | vLLM ports, one per entry in `model_paths` | `(8000 8001 ...)` |

**Input data:** Script uses `data/500_steps_filtered.json`.

**Run:**

```bash
cd AndroidControl

# 1) Edit plan_gen_guilibra.sh: set screenshot_dir, model_paths, exp_names, ports
# 2) Start vLLM (or compatible) on each port
# 3) Run:

bash plan_gen_guilibra.sh
```

Outputs: `data/${dir}/${exp_name}/plan_${level}_${index}_temperature${temperature}.jsonl` for `level ∈ {high, low}`.

**Evaluation (RL-style metrics):** Run `get_score_rl.sh` to compute step-level and best-of-n scores from generated plans. Set `sample_file` to your sample JSON (e.g. `data/500_steps_filtered.json` or the bbox-filtered file), `dir` to the directory that contains the plan files (e.g. `data/exp_name`), and `relative_coord` to `0` or `1` to match your model (Qwen3-based models use `1` and Qwen2.5-based models use `0`). The script runs `eval_rl.py` for best-of-1 and `eval_best_of_n.py` for best-of-4, for both `high` and `low` instruction levels. Expected files under `dir`: `plan_high_0_temperature0.0.jsonl`, `plan_high_0_temperature1.0.jsonl`, and the same for `low`.

```bash
cd AndroidControl
# Edit get_score_rl.sh: set sample_file=, dir=, relative_coord=
bash get_score_rl.sh
```

---

### See-Act-V Baselines

For gpt-based baselines, follow the original instruction in each folder to generate plans, and extract queries, run grounding inference, and run eval.py



## Code used for Generating Action Description and Filtering Evaluation Data 


`filter_eval_data.py` and `gen_action_description_MMM2web.py`




