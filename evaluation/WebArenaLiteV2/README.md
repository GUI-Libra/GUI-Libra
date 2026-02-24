# WebArena-Lite-v2 Benchmark Evaluation

WebArena-Lite-v2 is a benchmark deployed in [ScaleCUA](https://github.com/OpenGVLab/ScaleCUA), providing a framework designed specifically for evaluating pure visual GUI agents in web environments. As an improvement upon [WebArena-Lite](https://github.com/THUDM/VisualAgentBench), it offers 154 tasks across five website types, covering QA, page content matching, and more. We acknowledge the excellent contributions of the WebArena community.

This evaluation framework is adapted from [ScaleCUA](https://github.com/OpenGVLab/ScaleCUA/tree/main/evaluation/WebArenaLiteV2) with modifications to the environment setup based on [scripts from Maxime Gasse](https://github.com/gasse/webarena-setup/tree/main/webarena) for a more stable MAP environment.

## 🏗️ Environment Setup

Here we provide a quick local setup. For Docker-based setup, refer to [ScaleCUA](https://github.com/OpenGVLab/ScaleCUA) for more details.

```bash
git clone https://github.com/GUI-Libra/GUI-Libra
cd GUI-Libra/evaluation/WebArenaLiteV2

bash setup_env.sh
```

`setup_env.sh` will:
1. Create a Python 3.12 virtual environment and install dependencies
2. Download and load all Docker images (Shopping, Shopping Admin, Reddit, GitLab, Map)
3. Set `ARCHIVES_LOCATION` in `launcher/00_vars.sh` to the resolved absolute path

After setup, check or edit `launcher/00_vars.sh` to confirm your hostname and ports, then **test the environment** by running:

```bash
python launcher/start.py
```

If everything is correct, visit `http://localhost:{PORT}` (e.g., `http://localhost:7770`) to see the website.

## 🚀 Running the Evaluation

Supported models and their corresponding agent configs:

| Model | HF Path | vllm `--served-model-name` | Agent Config |
|-------|---------|----------------------|--------------|
| ScaleCUA | `OpenGVLab/ScaleCUA-7B` | `scalecua` | `scalecua_native_agent` |
| Qwen2.5-VL | `Qwen/Qwen2.5-VL-7B-Instruct` | `qwen25vl` | `qwen25vl_native_agent` |
| Qwen3-VL | `Qwen/Qwen3-VL-8B-Instruct` | `qwen3vl` | `qwen3vl_native_agent` |
| GUI-Libra | `GUILibra/GUILibra-8B-VL` | `guilibra` | `guilibra_native_agent_qwen3vl` |

**1. Configure the environment:**
Edit the following two files:

- `config/agent/<agent_config>.yaml`: Modify `base_url` and `model` to point to your deployed model endpoint.
- `config/env/web.yaml`: Update `explicitly_allowed_ports` to match the ports set in `launcher/00_vars.sh`. Other parameters generally do not need modification.

**2. Reinitialize the web environment:** (required before each evaluation run)

```bash
python launcher/start.py
```

**3. Serve the Model via vLLM:**

We use [vLLM](https://github.com/vllm-project/vllm) to deploy GUI-Libra and other baseline models. After installing vLLM, serve a model with:

```bash
python -m vllm.entrypoints.openai.api_server \
    --served-model-name <served_model_name> \
    --model <HF path> \
    --port 10028 \
    -tp 2
```

**4. Run the evaluation:**

```bash
export OPENAI_API_KEY="<your_api_key>"
export OPENAI_BASE_URL="<your_api_base_url>"

python agent_run.py \
    --env_config_path config/envs/web.yaml \
    --agent_config_path config/agent/<agent_config>.yaml \
    --task_config_path tasks/ \
    --num_workers 8 \
    --exp_name my_experiment \
    --max_steps 15
```

**4. Evaluation results** are saved under `results/{exp_name}/`:

```
results/{exp_name}/
├── results.jsonl              # overall results
└── {task_id}/
    ├── result.json            # task completion status
    └── trajectory/            # per-step screenshots
```

## 👍 Acknowledgements

This project is built upon the following projects. Thanks for their great work!
- [WebArena](https://github.com/web-arena-x/webarena)
- [VisualAgentBench (WebArena-Lite)](https://github.com/THUDM/VisualAgentBench)
- [Agent-S](https://github.com/simular-ai/Agent-S)
- [ScaleCUA](https://github.com/OpenGVLab/ScaleCUA)
- [Project from Maxime Gasse](https://github.com/gasse/webarena-setup/tree/main/webarena)
