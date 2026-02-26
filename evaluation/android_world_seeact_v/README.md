# AndroidWorld Evaluation

This guide explains how to set up the AndroidWorld Docker environment, serve your model, and run the evaluation.

## Overview

The evaluation pipeline consists of three parts:

1. **AndroidWorld Docker containers** — headless Android emulators exposed via HTTP APIs
2. **Model serving** — your GUI agent model served via an OpenAI-compatible API (e.g., vLLM)
3. **Evaluation script** — orchestrates task execution across multiple emulators in parallel

## Prerequisites

- Linux server with KVM support (`/dev/kvm` must be available)
- Docker and Docker Compose installed
- NVIDIA GPU(s) for model serving (if using local models)

## Step 1: Build the AndroidWorld Docker Image

Build the Docker image from the [official AndroidWorld repository](https://github.com/google-research/android_world):

```bash
git clone https://github.com/google-research/android_world.git
cd android_world
docker build -t android_world:latest .
```

This creates an image that bundles the Android emulator, ADB, and a FastAPI server exposing the environment over HTTP.

## Step 2: Launch Android Emulators

We provide `docker-compose.yml` and `nginx.conf` to run multiple Android emulators behind a single Nginx gateway. The default configuration launches **15 emulators** (emu1–emu15) on a single machine.

### Start the containers

```bash
cd evaluation/android_world_seeact_v
docker compose up -d
```

This will:
- Launch 15 Android emulator containers (`emu1`–`emu15`), each running a Pixel 6 AVD (API 33)
- Start an Nginx gateway on port `23333` that routes `/emu1/`, `/emu2/`, ..., `/emu15/` to each container

### Verify health

Wait a few minutes for the emulators to boot, then check:

```bash
# Check individual emulator health
curl http://localhost:23333/emu1/health

# Check all containers
docker compose ps
```

All emulators should show as `healthy` before proceeding.

### Reset emulators

If emulators become unresponsive or you need a clean restart:

```bash
bash reset_env.sh
```

This stops and removes all containers, brings them back up, and disables the pointer location overlay on each emulator.

### Scaling

To use fewer emulators, edit `docker-compose.yml` to comment out unneeded services and update the `gateway` `depends_on` list accordingly. Also remove the corresponding `/emuN/` blocks in `nginx.conf`.

## Step 3: Install the Evaluation Environment

```bash
conda create -n android_world python=3.11.8
conda activate android_world

cd evaluation/android_world_seeact_v
pip install -r requirements.txt
python setup.py install
```

You also need to install [AndroidEnv](https://github.com/google-deepmind/android_env):

```bash
git clone https://github.com/google-deepmind/android_env.git
cd android_env
python setup.py install
```

## Step 4: Serve Your Model

The evaluation script communicates with the model via an OpenAI-compatible API. We recommend using [vLLM](https://github.com/vllm-project/vllm):

```bash
python -m vllm.entrypoints.openai.api_server \
    --model GUI-Libra/GUI-Libra-8B \
    --port 8001 \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --limit-mm-per-prompt image=20 \
    --trust-remote-code
```

Note down the `base_url` (e.g., `http://localhost:8001/v1`) for the next step.

## Step 5: Run the Evaluation

### Edit the run script

Open `run.sh` and configure:

```bash
# Model to evaluate (HuggingFace model name or path)
model_name=GUI-Libra/GUI-Libra-8B

# Output directory
output_path='./eval_results/'${model_name}
save_img_dir=${output_path}/images

# vLLM API endpoint
base_url=http://localhost:8001/v1

# Emulator gateway host(s)
HOST1=http://localhost:23333
HOSTS=($HOST1)
```

The script automatically generates emulator URLs (`/emu1` through `/emu15`) from the host addresses.

### Run

```bash
bash run.sh
```

This calls `run_suite_on_docker_mp.py` which:
- Distributes the 116 AndroidWorld tasks across available emulators (we found only 115 tasks can be initilized.)
- Runs each task with the specified model as the agent
- Saves results and screenshots to the output directory

### Key arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model name (HuggingFace ID or local path) | `gpt-4o` |
| `--base_url` | OpenAI-compatible API endpoint | `http://localhost:8000/v1` |
| `--env_urls` | Comma-separated emulator URLs | `http://localhost:23333/emu1` |
| `--num_workers` | Number of parallel workers | `1` |
| `--max_steps` | Max agent steps per task | `20` |
| `--temperature` | LLM sampling temperature | `0.0` |
| `--output_path` | Directory to save results | `./test` |
| `--task_index` | Start from this task index (for resuming) | `-1` |

### Multi-host setup

To run emulators across multiple machines, add more hosts:

```bash
HOST1=http://server1:23333
HOST2=http://server2:23333
HOSTS=($HOST1 $HOST2)
```

Each host should have its own `docker-compose` deployment running 15 emulators, giving 30 parallel environments total.



## Files

| File | Description |
|------|-------------|
| `docker-compose.yml` | Docker Compose config for 15 Android emulators + Nginx gateway |
| `nginx.conf` | Nginx reverse proxy routing `/emuN/` to each emulator container |
| `reset_env.sh` | Script to stop, remove, and restart all emulator containers |
| `run.sh` | Main evaluation launch script |
| `run_suite_on_docker_mp.py` | Multi-process evaluation client |
| `run_suite_on_docker.py` | Single-process evaluation client |

## References

- [AndroidWorld](https://github.com/google-research/android_world) — Official environment repository
