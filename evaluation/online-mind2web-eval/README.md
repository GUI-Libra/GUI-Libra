# Online-Mind2Web Evaluation

Evaluation code for reproducing results on Online-Mind2Web benchmark.

## Setup

Install python dependencies:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

The browser relies on `browser-env` of WebArena:

```bash
git clone https://github.com/web-arena-x/webarena.git
cd webarena
pip install -e .
cd ..
playwright install chromium
```

Finally, for evaluating the agent trajectories, please refer to [Online-Mind2Web](https://github.com/OSU-NLP-Group/Online-Mind2Web) for setup.


## Evaluation

### Step 1: Serve the model

Use [vLLM](https://github.com/vllm-project/vllm) to serve the GUIPivot model checkpoint:

```bash
vllm serve GUI-Libra/GUI-Libra-8B \
    --port 20001 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9
```

### Step 2: Run the agent

```bash
python run.py \
    --tasks_path configs/mind2web.300.jsonl \
    --gpt.model GUI-Libra/GUI-Libra-8B \
    --gpt.openai_api_base http://localhost:20001/v1 \
    --gpt.openai_api_key token-abc123 \
    --num_processes 4
```

The trajectories will be saved in `outputs` folder by default.

### Step 3: Eval on the trajectories
Please refer to [Online-Mind2Web](https://github.com/OSU-NLP-Group/Online-Mind2Web) for getting the evaluation scores.


