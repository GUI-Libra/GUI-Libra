# SFT 训练指南

本文档说明如何使用本目录进行 GUI 智能体的监督微调（SFT）训练，基于 Qwen2.5-VL / Qwen3-VL 视觉语言模型。

## 支持模型

| 模型系列 | 脚本 | 支持规格 |
|----------|------|----------|
| Qwen2.5-VL | `scripts/train_qwen2_5.sh` | 3B, 7B |
| Qwen3-VL | `scripts/train_qwen3.sh` | 4B, 8B |

## 环境安装

### 方式一：使用 setup.sh（推荐）

```bash
bash setup.sh
```

### 方式二：手动安装

```bash
conda env create -n py310 python=3.10
conda activate py310
pip install -r requirements.txt
pip uninstall torch torchvision torchaudio
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn==2.8.3 --no-build-isolation
pip install -e .
pip install transformers==4.57.3
```

### 可选：W&B 日志

若使用 `--report_to "wandb"`，需先登录：

```bash
wandb login
# 或 export WANDB_API_KEY=your_key
```

## 数据准备

### 1. 设置环境变量

训练前需设置数据根目录：

```bash
export DATA_ROOT=/path/to/your/datasets
```

若图像与标注不在同一目录，可在脚本中修改 `IMAGE_FOLDER`（默认 `./data/images`）。

### 2. 目录结构

数据需按以下结构组织：

```
${DATA_ROOT}/
└── annotations/                          # JSON 标注文件
    ├── mind2web-reasoning_and_grounding_changecoord.json
    ├── guiact-web-reasoning_and_grounding_changecoord.json
    ├── guiact-web-chinese-reasoning_and_grounding_changecoord.json
    ├── coat-terminal-reasoning_and_grounding_changecoord.json
    ├── amex-reasoning_and_grounding_changecoord.json
    ├── aitw-reasoning_and_grounding_changecoord.json
    ├── android_control-reasoning_and_grounding_changecoord.json
    └── gui-odyssey-reasoning_and_grounding_changecoord.json

./data/images/                            # 图像目录（由 IMAGE_FOLDER 指定）
├── mind2web/
├── guiact-web-multi-v2/images/
├── guiact-web-multi-v2-chinese/images/
├── android_in_the_zoo/train/
├── amex/images/
├── aitw-v1/images/
├── android_control/images/
└── gui-odyssey/images/
```

### 3. 数据配置 YAML

`data/` 目录下提供多种训练模式对应的 YAML 配置：

| 文件 | 说明 |
|------|------|
| `reasoning_and_grounding_changecoord.yaml` | 仅 reasoning + grounding |
| `reasoning_and_grounding_changecoord_noreason.yaml` | 仅 grounding（无推理） |
| `reasoning_and_grounding_changecoord_mixnoreasoning.yaml` | 混合 reasoning 与 noreason |
| `reasoning_and_grounding_changecoord_mixnoreasoning_qwen3.yaml` | Qwen3 专用，含 _1000 坐标格式 |

YAML 中的 `json_path` 使用 `${DATA_ROOT}`，运行时会自动展开。

## 训练

### Qwen2.5-VL 训练

```bash
export DATA_ROOT=/path/to/your/datasets

bash scripts/train_qwen2_5.sh
```

**可配置项**（在脚本中修改）：
- `llm_index`：模型索引（0=3B, 1=7B）
- `mode`：训练模式，对应 `data/${mode}.yaml`
- `use_action_weight`：是否对 action token 加权
- `action_weight`：action 权重系数

**Epoch 规则**：若 `mode` 不含 `mixnoreasoning`，`num_epochs` 自动设为 2，以保证训练步数一致。

### Qwen3-VL 训练

```bash
export DATA_ROOT=/path/to/your/datasets

bash scripts/train_qwen3.sh
```

**可配置项**：
- `llm_index`：0=4B, 1=8B
- `mode`：默认 `reasoning_and_grounding_changecoord_mixnoreasoning_qwen3`

### 输出

-  checkpoint 保存到 `checkpoints/${SFT_RUN_NAME}/`
- 默认每 100 步保存一次，最多保留 5 个 checkpoint

## 主要训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `per_device_train_batch_size` | 4 | 每卡 batch size |
| `gradient_accumulation_steps` | 8 | 梯度累积步数 |
| `learning_rate` | 1e-5 | 学习率 |
| `model_max_length` | 24576 | 最大序列长度 |
| `freeze_visual_encoder` | False | 是否冻结视觉编码器 |
| `--num_processes` | 8 | 使用的 GPU 数量 |

## 多机训练

多机时需设置：

```bash
export MASTER_ADDR=<主节点 IP>
export MASTER_PORT=29504
```

并在各节点正确配置 `RANK`、`WORLD_SIZE` 等环境变量（或使用 `accelerate config`）。

