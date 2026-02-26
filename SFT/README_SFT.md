# SFT Training

This directory contains the code for action-aware supervised fine-tuning (SFT) of GUI agents based on Qwen2.5-VL and Qwen3-VL.

## Supported Models

| Model | Script | Sizes |
|-------|--------|-------|
| Qwen2.5-VL | `scripts/train_qwen2_5.sh` | 3B, 7B |
| Qwen3-VL | `scripts/train_qwen3.sh` | 4B, 8B |

## Installation

```bash
bash setup.sh
```

## Data Preparation

### 1. Download the Dataset

Clone the [GUI-Libra-81K-SFT](https://huggingface.co/datasets/GUI-Libra/GUI-Libra-81K-SFT) dataset from Hugging Face:

```bash
git lfs install
git clone https://huggingface.co/datasets/GUI-Libra/GUI-Libra-81K-SFT
cd GUI-Libra-81K-SFT
```

After cloning, you will see:
- `data/annotations/` — JSON annotation files
- `data/images/` — split image archives (`*.tar.gz.part-*`)

### 2. Merge and Extract Images

The image archives are split into parts and need to be merged before extraction:

```bash
cd data/images
mkdir -p ../images_extracted

# Merge split parts and extract
for base in $(ls *.tar.gz.part-* | sed -E 's/\.part-[0-9]+$//' | sort -u); do
  name="${base%.tar.gz}"
  echo "[MERGE + EXTRACT] ${base}.part-* -> ../images_extracted/${name}"
  cat "${base}".part-* | tar -xzf - -C ../images_extracted/
done
```

To extract only a specific subset (e.g., `gui-odyssey`):

```bash
cat gui-odyssey.tar.gz.part-* | tar -xzf - -C ../images_extracted/
```

### 3. Place Data Under the SFT Directory

Copy or symlink the downloaded data into the SFT directory:

```bash
# Assuming you are in the GUI-Libra/SFT directory
# Option 1: Symlink (recommended, saves disk space)
ln -s /path/to/Libra-81K-SFT/data/annotations ./data/annotations
ln -s /path/to/Libra-81K-SFT/data/images_extracted ./data/images

# Option 2: Copy
cp -r /path/to/Libra-81K-SFT/data/annotations ./data/annotations
cp -r /path/to/Libra-81K-SFT/data/images_extracted ./data/images
```

The expected directory structure under `SFT/`:

```
SFT/
├── data/
│   ├── annotations/                              # JSON annotation files
│   │   ├── mind2web-reasoning_and_grounding_changecoord.json
│   │   ├── guiact-web-reasoning_and_grounding_changecoord.json
│   │   ├── guiact-web-chinese-reasoning_and_grounding_changecoord.json
│   │   ├── coat-terminal-reasoning_and_grounding_changecoord.json
│   │   ├── amex-reasoning_and_grounding_changecoord.json
│   │   ├── aitw-reasoning_and_grounding_changecoord.json
│   │   ├── android_control-reasoning_and_grounding_changecoord.json
│   │   ├── gui-odyssey-reasoning_and_grounding_changecoord.json
│   │   └── ... (noreason variants, _1000 variants for Qwen3)
│   ├── images/                                    # Extracted image folders
│   │   ├── mind2web/
│   │   ├── guiact-web-multi-v2/images/
│   │   ├── guiact-web-multi-v2-chinese/images/
│   │   ├── android_in_the_zoo/train/
│   │   ├── amex/images/
│   │   ├── aitw-v1/images/
│   │   ├── android_control/images/
│   │   └── gui-odyssey/images/
│   ├── reasoning_and_grounding_changecoord.yaml
│   ├── reasoning_and_grounding_changecoord_mixnoreasoning.yaml
│   └── reasoning_and_grounding_changecoord_mixnoreasoning_qwen3.yaml
├── scripts/
│   ├── train_qwen2_5.sh
│   └── train_qwen3.sh
└── ...
```

## Training

### Configuration

Before running, edit the training script to adjust:

| Variable | Description | Default |
|----------|-------------|---------|
| `llm_index` | Model size index (Qwen2.5-VL: 0=3B, 1=7B; Qwen3-VL: 0=4B, 1=8B) | `0` |
| `IMAGE_FOLDER` | Path to the extracted images directory | `./data/images` |
| `DATA_ROOT` | Root path for annotation files (env variable used in YAML configs) | — |
| `use_action_weight` | Enable action-aware token weighting | `True` (Qwen2.5-VL) / `False` (Qwen3-VL) |
| `action_weight` | Weight multiplier for action tokens | `2.0` |
| `--num_processes` | Number of GPUs | `8` |

### Run Qwen2.5-VL SFT

```bash
cd SFT

# Set DATA_ROOT to where annotations are stored
# If you symlinked annotations to ./data/annotations, use:
export DATA_ROOT=./data

bash scripts/train_qwen2_5.sh
```

The script trains on two modes by default:
- `reasoning_and_grounding_changecoord_mixnoreasoning` — mixed reasoning + direct-action data (1 epoch)
- `reasoning_and_grounding_changecoord` — reasoning-only data (2 epochs, automatically doubled to match training steps)

### Run Qwen3-VL SFT

```bash
cd SFT
export DATA_ROOT=./data

bash scripts/train_qwen3.sh
```

This trains on `reasoning_and_grounding_changecoord_mixnoreasoning_qwen3` mode with coordinates in [0, 1000] format.

### Output

Checkpoints are saved to `checkpoints/<run_name>/`, with a checkpoint saved every 100 steps (max 5 retained).

## Key Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `per_device_train_batch_size` | 4 | Batch size per GPU |
| `gradient_accumulation_steps` | 8 | Gradient accumulation |
| `learning_rate` | 1e-5 | Learning rate |
| `model_max_length` | 24576 | Max sequence length |
| `freeze_visual_encoder` | False | Train the vision encoder |
| `attn_implementation` | flash_attention_2 | Attention backend |

## Multi-Node Training

For multi-node setups, set the following environment variables before launching:

```bash
export MASTER_ADDR=<master-node-ip>
export MASTER_PORT=29504
```

And configure `RANK`, `WORLD_SIZE` via `accelerate config` or environment variables on each node.
