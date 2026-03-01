<p align="center">
  <img src="images/logo.png" width="200" alt="GUI-Libra Logo">
</p>


<h2 align="center">GUI-Libra: Training Native GUI Agents to Reason and Act<br>with Action-aware Supervision and Partially Verifiable RL</h2>

<p align="center">
  <a href="https://gui-libra.github.io/"><img src="https://img.shields.io/badge/🌐_Project_Page-blue" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2602.22190"><img src="https://img.shields.io/badge/📄_Paper-red" alt="Paper"></a>
  <a href="https://github.com/GUI-Libra/GUI-Libra"><img src="https://img.shields.io/badge/💻_Code-green" alt="Code"></a>
  <a href="https://huggingface.co/GUI-Libra"><img src="https://img.shields.io/badge/🤗_Models-yellow" alt="Models&Datasets"></a>
</p>

<p align="center">
  <a href="https://yangrui2015.github.io/">Rui Yang</a><sup>1</sup>,
  <a href="https://qianhuiwu.github.io/">Qianhui Wu</a><sup>2</sup>,
  <a href="https://zhaoyang.win/">Zhaoyang Wang</a><sup>3</sup>,
  <a href="https://jeremy-chy.github.io/">Hanyang Chen</a><sup>1</sup>,
  <a href="https://empathyang.github.io/">Ke Yang</a><sup>1</sup>,
  <a href="https://sites.google.com/site/hcheng2site/Home">Hao Cheng</a><sup>2</sup>,
  <br>
  <a href="https://www.huaxiuyao.io/">Huaxiu Yao</a><sup>3</sup>,
  <a href="https://scholar.google.com/citations?user=u1CNjgwAAAAJ&hl=zh-CN">Baolin Peng</a><sup>2</sup>,
  <a href="https://www.huan-zhang.com/">Huan Zhang</a><sup>1</sup>,
  <a href="https://www.microsoft.com/en-us/research/people/jfgao/">Jianfeng Gao</a><sup>2</sup>,
  <a href="https://tongzhang-ml.org/">Tong Zhang</a><sup>1</sup>
  <br>
  <sup>1</sup>UIUC &nbsp; <sup>2</sup>Microsoft &nbsp; <sup>3</sup>UNC-Chapel Hill
</p>

---

## Overview

**GUI-Libra** is a post-training framework that turns open-source VLMs into strong native GUI agents — models that see a screenshot, think step-by-step, and output an executable action, all within a single forward pass.

We find that naively adding chain-of-thought (CoT) to GUI agents *hurts* grounding accuracy, and that standard RLVR-style training cannot achieve stable offline-to-online performance because GUI rewards are only *partially* verifiable. GUI-Libra solves both:

| Component | What it does |
|-----------|-------------|
| **GUI-Libra-81K** | 81K-step reasoning dataset with action re-prediction filtering and bounding-box coordinate verification |
| **Action-Aware SFT** | Mixes reasoning and direct-action data; reweights tokens so the model doesn't forget *where to click* while learning *why to click* |
| **Conservative RL** | KL-regularized GRPO that stays stable under ambiguous rewards, with success-adaptive scaling to tame noisy negative gradients |

The result: **GUI-Libra-4B/8B match or outperform GPT-4o/GPT-4.1/GPT-5-mini and 72/32B native models** on AndroidWorld, WebArena-Lite-v2, and Online-Mind2Web — without any online data collection.


## To Do List

- [x] Release training code (SFT + RL, supporting both Qwen2.5-VL and Qwen3-VL models)
- [x] Release evaluation code (WebArena-Lite-v2, Online-Mind2Web)
- [x] Release GUI-Libra-81K dataset
- [x] Release model checkpoints (GUI-Libra-3B/4B/7B/8B)
- [x] AndroidWorld evaluation code
- [x] Offline evaluation code (MM-Mind2Web, AndroidControl)


## Training Pipeline

GUI-Libra follows a two-stage post-training pipeline:

```
Base VLM ──► Action-Aware SFT (ASFT) ──► Conservative RL (GRPO) ──► GUI-Libra
```

### Stage 1: Action-Aware Supervised Fine-Tuning


See [`SFT/README.md`](SFT/README.md) for full training instructions.

### Stage 2: Reinforcement Learning with Partial Verifiable Rewards

See [`EasyR1/README.md`](EasyR1/README.md) for full RL training instructions.

## Project Structure

```
GUI-Libra/
├── SFT/                          # Supervised fine-tuning
│   ├── train.py                  # Main training script
│   ├── src/aguvis/               # Dataset, trainer, constants
│   ├── scripts/                  # Training shell scripts
│   │   ├── train_qwen2_5.sh      # Qwen2.5-VL (3B/7B)
│   │   └── train_qwen3.sh        # Qwen3-VL (4B/8B)
│   ├── data/                     # Data config YAMLs
│   └── README.md                  # SFT documentation
│
├── EasyR1/                       # Reinforcement learning (based on EasyR1/veRL)
│   ├── verl/                     # RL training framework
│   ├── examples/
│   │   ├── gui_grpo.sh           # Qwen2.5-VL GRPO training
│   │   ├── gui_grpo_qwen3.sh     # Qwen3-VL GRPO training
│   │   ├── reward_function/      # GUI reward functions
│   │   └── README.md             # RL documentation
│   └── README.md                 # EasyR1 framework docs
│
├── evaluation/                   # Evaluation benchmarks
│   ├── WebArenaLiteV2/           # WebArena-Lite-v2 evaluation
│   ├── online-mind2web-eval/     # Online-Mind2Web evaluation
│   ├── android_world_seeact_v/   # AndroidWorld evaluation
│   └── offline_evaluation/        # Offline (MM-Mind2Web, AndroidControl)
│
└── images/                       # Project assets
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/GUI-Libra/GUI-Libra.git
cd GUI-Libra
```

### 2. SFT Training

```bash
cd SFT
bash setup.sh                                    # install dependencies
export DATA_ROOT=/path/to/your/datasets           # set data root
bash scripts/train_qwen2_5.sh                     # train Qwen2.5-VL
# or
bash scripts/train_qwen3.sh                       # train Qwen3-VL
```

### 3. RL Training

```bash
cd EasyR1
pip install -e .
# Edit examples/gui_grpo.sh to set MODEL_PATH, TRAIN_FILES, VAL_FILES
bash examples/gui_grpo.sh                         # Qwen2.5-VL GRPO
# or
bash examples/gui_grpo_qwen3.sh                   # Qwen3-VL GRPO
```

### 4. Evaluation

#### WebArena-Lite-v2

```bash
cd evaluation/WebArenaLiteV2
bash setup_env.sh                                 # set up Docker environments
python launcher/start.py                          # start web environments
# Serve model via vLLM, then:
python agent_run.py \
    --platform web \
    --env_config_path config/env/web.yaml \
    --agent_config_path config/agent/<agent_config>.yaml \
    --task_config_path tasks/ \
    --num_workers 8 \
    --max_steps 15
```

#### Online-Mind2Web

```bash
cd evaluation/online-mind2web-eval
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e .
# Serve model via vLLM, then:
python run.py \
    --tasks_path configs/mind2web.300.jsonl \
    --gpt.model <model_name> \
    --gpt.openai_api_base http://localhost:20001/v1 \
    --num_processes 4
```

#### AndroidWorld

```bash
cd evaluation/android_world_seeact_v
# Launch 15 Android emulators via Docker
docker compose up -d
# Serve model via vLLM, then edit run.sh and run:
bash run.sh
```

See [`evaluation/android_world_seeact_v/README.md`](evaluation/android_world_seeact_v/README.md) for detailed setup instructions.

#### Offline evaluation (MM-Mind2Web, AndroidControl)

```bash
cd evaluation/offline_evaluation
# See README for data download and plan_gen_guilibra.sh usage
```

See [`evaluation/offline_evaluation/README.md`](evaluation/offline_evaluation/README.md) for data setup, planning scripts, and evaluation pipelines.

## Data Format

Each training sample follows a unified structured format:

**Input**: system prompt + user instruction + interaction history + screenshot

**Output**:
```
<think>
Reasoning about the current UI state, reflecting on progress,
and planning the next action...
</think>
<answer>
{
  "action_description": "brief description of the action",
  "action_type": "Click",
  "value": "None",
  "point_2d": [x, y]
}
</answer>
```
> [!NOTE]
> We use `<thinking></thinking>` for Qwen3-based models instead of `<think></think>`.

Supported action types: `Click`, `Write`, `Terminate`, `Swipe`, `Scroll`, `NavigateHome`, `Answer`, `Wait`, `OpenAPP`, `NavigateBack`, `KeyboardPress`, `LongPress`, `Select`.

## Citation

```bibtex
@misc{yang2026guilibratrainingnativegui,
      title={GUI-Libra: Training Native GUI Agents to Reason and Act with Action-aware Supervision and Partially Verifiable RL}, 
      author={Rui Yang and Qianhui Wu and Zhaoyang Wang and Hanyang Chen and Ke Yang and Hao Cheng and Huaxiu Yao and Baoling Peng and Huan Zhang and Jianfeng Gao and Tong Zhang},
      year={2026},
      eprint={2602.22190},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.22190}, 
}
```

## Acknowledgements

This project builds upon the following excellent work:

- [EasyR1](https://github.com/hiyouga/EasyR1) — RL training framework
- [AGUVIS](https://github.com/xlang-ai/aguvis) — GUI agent framework and data
- [ScaleCUA](https://github.com/OpenGVLab/ScaleCUA) — WebArena-Lite-v2 evaluation
- [WebArena](https://github.com/web-arena-x/webarena) — Web environment
- [Online-Mind2Web](https://github.com/OSU-NLP-Group/Online-Mind2Web) — Online evaluation benchmark
- [UGround](https://github.com/OSU-NLP-Group/UGround) — Evaluation on MM-Mind2Web, AndroidControl, and AndroidWorld

## License

This project is released under the [MIT License](LICENSE).
