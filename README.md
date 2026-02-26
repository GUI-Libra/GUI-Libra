<p align="center">
  <img src="images/logo.png" width="400" alt="GUI-Libra Logo">
</p>

<h1 align="center">GUI-Libra</h1>

<p align="center">
  <strong>Training Native GUI Agents to Reason and Act with Action-aware Supervision and Partially Verifiable RL</strong>
</p>

<p align="center">
  <a href="https://gui-libra.github.io/">Project Page</a> &bull;
  <a href="https://gui-libra.github.io/">Paper</a> &bull;
  <a href="https://github.com/GUI-Libra/GUI-Libra">Code</a> &bull;
  Models &bull;
  Dataset
</p>

<p align="center">
  Rui Yang<sup>1</sup>, Qianhui Wu<sup>2</sup>, Zhaoyang Wang<sup>3</sup>, Hanyang Chen<sup>1</sup>, Ke Yang<sup>1</sup>, Hao Cheng<sup>2</sup>, Huaxiu Yao<sup>3</sup>, Baolin Peng<sup>2</sup>, Huan Zhang<sup>1</sup>, Jianfeng Gao<sup>2</sup>, Tong Zhang<sup>1</sup>
  <br>
  <sup>1</sup>UIUC &nbsp; <sup>2</sup>Microsoft &nbsp; <sup>3</sup>UNC-Chapel Hill
</p>

---

## Overview

Open-source native GUI agents have made rapid progress in visual grounding and low-level action execution, yet they still lag behind closed-source systems on long-horizon navigation tasks that demand both high-level reasoning and precise actions. **GUI-Libra** addresses two key limitations: (1) the scarcity of high-quality, action-aligned reasoning data, and (2) the direct adoption of generic post-training pipelines that overlook challenges unique to GUI agents.


GUI-Libra tackles these with a tailored recipe:
1. **81K Curated Dataset** — action-aligned reasoning data with agreement filtering and coordinate verification
2. **Action-Aware SFT (ASFT)** — mixed reasoning/direct-action supervision with token-level reweighting to preserve grounding under long CoT
3. **Conservative RL** — KL-regularized GRPO with success-adaptive negative gradient scaling for stable training under partial verifiability



## Training Pipeline

GUI-Libra follows a two-stage post-training pipeline:

```
Base VLM ──► Action-Aware SFT (ASFT) ──► Conservative RL (GRPO) ──► GUI-Libra
```

### Stage 1: Action-Aware Supervised Fine-Tuning

- **Mixed supervision**: trains on both reasoning-then-action and direct-action data
- **Token reweighting**: assigns higher weights to action and grounding tokens (controlled by α_a and α_g), reducing grounding degradation caused by long CoT
- **Supported models**: Qwen2.5-VL (3B/7B), Qwen3-VL (4B/8B)

See [`SFT/README_SFT.md`](SFT/README_SFT.md) for full training instructions.

### Stage 2: Reinforcement Learning with Partial Verifiable Rewards

- **GRPO with KL regularization**: constrains policy drift under reward ambiguity, improving offline-to-online predictability (Pearson r from 0.63 to 0.89)
- **Success-adaptive negative gradient scaling**: downweights unreliable negative gradients from ambiguously "wrong" actions
- **Weighted positive-negative advantage estimation**: balances learning from successful and failed rollouts

See [`EasyR1/README_GUI_RL.md`](EasyR1/README_GUI_RL.md) for full RL training instructions.

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
│   └── README_SFT.md             # SFT documentation
│
├── EasyR1/                       # Reinforcement learning (based on EasyR1/veRL)
│   ├── verl/                     # RL training framework
│   ├── examples/
│   │   ├── gui_grpo.sh           # Qwen2.5-VL GRPO training
│   │   ├── gui_grpo_qwen3.sh     # Qwen3-VL GRPO training
│   │   ├── reward_function/      # GUI reward functions
│   │   └── README_GUI_RL.md      # RL documentation
│   └── README.md                 # EasyR1 framework docs
│
├── evaluation/                   # Evaluation benchmarks
│   ├── WebArenaLiteV2/           # WebArena-Lite-v2 evaluation
│   └── online-mind2web-eval/     # Online-Mind2Web evaluation
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

Supported action types: `Click`, `Write`, `Terminate`, `Swipe`, `Scroll`, `NavigateHome`, `Answer`, `Wait`, `OpenAPP`, `NavigateBack`, `KeyboardPress`, `LongPress`, `Select`.

## Citation

```bibtex
@article{yang2025guilibra,
  title={GUI-Libra: Training Native GUI Agents to Reason and Act with Action-aware Supervision and Partially Verifiable RL},
  author={Yang, Rui and Wu, Qianhui and Wang, Zhaoyang and Chen, Hanyang and Yang, Ke and Cheng, Hao and Yao, Huaxiu and Peng, Baolin and Zhang, Huan and Gao, Jianfeng and Zhang, Tong},
  journal={arXiv preprint},
  year={2025}
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
