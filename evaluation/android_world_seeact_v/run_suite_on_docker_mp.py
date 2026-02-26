#!/usr/bin/env python3
"""
Multi-process evaluation client for Android World over HTTP.

Examples:
  python mp_eval.py \
    --env_urls http://localhost:23333/emu1,http://localhost:23333/emu2 \
    --num_workers 2 --model gpt-4o --max_steps 15
"""
import os
import json
import logging
import time
import argparse
from typing import Any, List, Tuple, Dict
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pydantic
import requests

# Android World imports (same as your single-process script)
from android_world.agents import infer
from android_world.env import json_action
from android_world.agents import seeact_v, seeact_v_gpt_noselfsummary, seeact_v_guilibra, seeact_v_local_docker, seeact_v_aguvis, \
     seeact_v_uitars, seeact_v_glm, seeact_v_qwen, seeact_v_qwen_step_summary

# ---------- Logging (console with msec) ----------
logger = logging.getLogger("mp_eval")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _console = logging.StreamHandler()
    _console.setFormatter(logging.Formatter(
        "[%(asctime)s.%(msecs)03d][%(processName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_console)

Params = Dict[str, int | str]


def get_args():
    parser = argparse.ArgumentParser(description="Android World MP Agent Config")

    # Selection/sharding
    parser.add_argument(
        "--task_index",
        type=int,
        default=-1,
        help="Start from this index in the global expanded (task_name, idx) list.",
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=-1,
        help="If >0, restrict to this many tasks (after task_index).",
    )

    # Output / run config
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.expanduser("./test"),
        help="Path to save logs/outputs.",
    )
    parser.add_argument(
        "--save_img_dir",
        type=str,
        default=os.path.expanduser("./test_imgs"),
        help="Path to save logs/outputs.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=15,
        help="Max agent steps per task.",
    )
    parser.add_argument(
        "--agent_name",
        type=str,
        default="seeact_v",
        help="Agent name (kept for compatibility).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM name for the agent.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for the LLM.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for LLM API.",
    )
    # Parallelism / environments
    parser.add_argument(
        "--env_urls",
        type=str,
        default="http://localhost:23333/emu1",
        help="Comma-separated list of base URLs to emulator gateways "
             "(e.g., http://localhost:23333/emu1,http://localhost:23333/emu2).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes to launch (usually equals number of env_urls).",
    )

    # Suite config mirrors your single-process behavior
    parser.add_argument(
        "--suite_max_index",
        type=int,
        default=-1,
        help="Passed to /suite/task_list as max_index.",
    )
    parser.add_argument(
        "--reinit_n_task_combinations",
        type=int,
        default=1,
        help="Suite reinitialize: n_task_combinations.",
    )
    parser.add_argument(
        "--reinit_seed",
        type=int,
        default=42,
        help="Suite reinitialize: seed.",
    )
    parser.add_argument(
        "--reinit_task_family",
        type=str,
        default="android_world",
        help="Suite reinitialize: task_family.",
    )

    return parser.parse_args()


# ---------- API models & client ----------

class Response(pydantic.BaseModel):
    status: str
    message: str


@dataclass
class AndroidEnvClient:
    base_url: str

    def reset(self, go_home: bool) -> Response:
        r = requests.post(f"{self.base_url}/reset", params={"go_home": go_home})
        r.raise_for_status()
        return Response(**r.json())

    def get_screenshot(self, wait_to_stabilize: bool = False) -> np.ndarray:
        r = requests.get(
            f"{self.base_url}/screenshot",
            params={"wait_to_stabilize": wait_to_stabilize},
        )
        r.raise_for_status()
        image = r.json()
        return np.array(image["pixels"])

    def execute_action(self, action: json_action.JSONAction) -> Response:
        payload = json.loads(action.json_str())
        r = requests.post(f"{self.base_url}/execute_action", json=payload)
        r.raise_for_status()
        return Response(**r.json())

    def get_suite_task_list(self, max_index: int) -> List[str]:
        r = requests.get(f"{self.base_url}/suite/task_list", params={"max_index": max_index})
        r.raise_for_status()
        return r.json()["task_list"]

    def get_suite_task_length(self, task_type: str) -> int:
        r = requests.get(f"{self.base_url}/suite/task_length", params={"task_type": task_type})
        r.raise_for_status()
        return r.json()["length"]

    def reinitialize_suite(self, n_task_combinations: int, seed: int, task_family: str) -> Response:
        r = requests.get(
            f"{self.base_url}/suite/reinitialize",
            params={
                "n_task_combinations": n_task_combinations,
                "seed": seed,
                "task_family": task_family,
            },
        )
        r.raise_for_status()
        return Response(**r.json())

    def initialize_task(self, task_type: str, task_idx: int) -> Response:
        params: Params = {"task_type": task_type, "task_idx": task_idx}
        r = requests.post(f"{self.base_url}/task/initialize", params=params)
        r.raise_for_status()
        return Response(**r.json())

    def tear_down_task(self, task_type: str, task_idx: int) -> Response:
        params: Params = {"task_type": task_type, "task_idx": task_idx}
        r = requests.post(f"{self.base_url}/task/tear_down", params=params)
        r.raise_for_status()
        return Response(**r.json())

    def get_task_score(self, task_type: str, task_idx: int) -> float:
        params: Params = {"task_type": task_type, "task_idx": task_idx}
        r = requests.get(f"{self.base_url}/task/score", params=params)
        r.raise_for_status()
        return r.json()["score"]

    def get_task_goal(self, task_type: str, task_idx: int) -> str:
        params: Params = {"task_type": task_type, "task_idx": task_idx}
        r = requests.get(f"{self.base_url}/task/goal", params=params)
        r.raise_for_status()
        return r.json()["goal"]

    def get_task_template(self, task_type: str, task_idx: int) -> str:
        params: Params = {"task_type": task_type, "task_idx": task_idx}
        r = requests.get(f"{self.base_url}/task/template", params=params)
        r.raise_for_status()
        return r.json()["template"]

    def close(self) -> None:
        r = requests.post(f"{self.base_url}/close")
        r.raise_for_status()

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            r.raise_for_status()
            return True
        except Exception as e:
            logger.info(f"[{self.base_url}] Environment is not healthy: {e}")
            return False


# ---------- Core eval routines ---------- 


def build_agent(client: AndroidEnvClient, model_name: str, temperature: float, base_url: str, file_logger: logging.Logger | None = None, save_img_dir: str | None=None):
    # Reuse original branching
    if 'ray2333' in model_name.lower() or 'gui-libra' in model_name.lower():
        # no grounding model for this
        return seeact_v_guilibra.SeeAct_V(
            client,
            infer.Gpt4Wrapper(model_name, temperature=temperature, base_url=base_url),
            file_logger=file_logger,
            save_img_dir=save_img_dir,
        )
    elif 'qwen' in model_name.lower():
        if 'step-summary' in model_name.lower():
            # replace step-summary with '' and call seeact_v_qwen_step_summary
            model_name = model_name.replace('step-summary', '')
            return seeact_v_qwen_step_summary.SeeAct_V(
                client,
                infer.Gpt4Wrapper(model_name, temperature=temperature, base_url=base_url),
                file_logger=file_logger,
                save_img_dir=save_img_dir,
            )
        else:
            return seeact_v_qwen.SeeAct_V(
                client,
                infer.Gpt4Wrapper(model_name, temperature=temperature, base_url=base_url),
                file_logger=file_logger,
                save_img_dir=save_img_dir,
            )
    elif any(k in model_name.lower() for k in ("gpt", "4o", "4.1", "o4")):
        if 'gpt-5' in model_name.lower() or 'o4' in model_name.lower():
            temperature = 1.0
        return seeact_v_gpt_noselfsummary.SeeAct_V(
            client,
            infer.Gpt4Wrapper(model_name, temperature=temperature, base_url=base_url),
            grounding_model_name="osunlp/UGround-V1-7B",
            file_logger=file_logger,
            save_img_dir=save_img_dir,
        )
    elif 'glm' in model_name.lower():
        return seeact_v_glm.SeeAct_V(
            client,
            infer.Gpt4Wrapper(model_name, temperature=temperature, base_url=base_url),
            grounding_model_name="osunlp/UGround-V1-7B",
            file_logger=file_logger,
            save_img_dir=save_img_dir,
        )
    elif 'aguvis' in model_name.lower():
        return seeact_v_aguvis.SeeAct_V(
            client,
            infer.Gpt4Wrapper(model_name, temperature=temperature, base_url=base_url),
            grounding_model_name="osunlp/UGround-V1-7B",
            file_logger=file_logger,
            save_img_dir=save_img_dir,
        )
    elif 'tars' in model_name.lower() and 'bytedance' in model_name.lower():
        return seeact_v_uitars.SeeAct_V(
            client,
            infer.Gpt4Wrapper(model_name, temperature=temperature, base_url=base_url),
            grounding_model_name="osunlp/UGround-V1-7B",
            file_logger=file_logger,
            save_img_dir=save_img_dir,
        )
    else:
        return seeact_v_local_docker.SeeAct_V(
            client,
            infer.Gpt4Wrapper(model_name, temperature=temperature, base_url=base_url),
            grounding_model_name="osunlp/UGround-V1-7B",
            file_logger=file_logger,
            save_img_dir=save_img_dir,
        )


def wait_until_healthy(client: AndroidEnvClient, max_wait_s: int = 300, interval_s: float = 1.0):
    start = time.time()
    while time.time() - start < max_wait_s:
        if client.health():
            return True
        time.sleep(interval_s)
    return False


def eval_one_task(client: AndroidEnvClient,
                  agent,
                  task_name: str,
                  true_task_id: int,
                  cur_idx: int,
                  max_steps: int,
                  log: logging.Logger | None = None) -> Tuple[bool, float, str]:
    """
    Time-instrumented single task run.
    Returns (agent_successful, task_score, message)
    """
    if log is None:
        log = logger

    # Mark start time
    wall_start = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    t0 = time.perf_counter()
    log.info(f"START {true_task_id} {task_name}#{cur_idx} at {wall_start}")

    client.reset(go_home=True)
    task_template = client.get_task_template(task_type=task_name, task_idx=cur_idx)
    if not ("{" in task_template and "}" in task_template) and cur_idx > 0:
        dt = time.perf_counter() - t0
        wall_end = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log.info(f"SKIP  {task_name}#{cur_idx} at {wall_end} duration={dt:.2f}s "
                 f"(non-parameterized template for idx>0)")
        return False, 0.0, "Template not parameterized for idx>0; stopping this task family."

    task_goal = client.get_task_goal(task_type=task_name, task_idx=cur_idx)
    agent.reset(task_id=true_task_id, repeat_id=cur_idx)
    log.info(f"Task goal: {task_goal}")

    try:
        client.initialize_task(task_type=task_name, task_idx=cur_idx)

        is_done = False
        steps = 0
        for steps in range(1, max_steps + 1):
            response = agent.step(task_goal)
            if getattr(response, "done", False):
                is_done = True
                break

        task_score = client.get_task_score(task_type=task_name, task_idx=cur_idx)
        agent_successful = is_done and task_score == 1
        msg = f'{"Task Successful ✅" if agent_successful else "Task Failed ❌"}; {task_goal} (score={task_score})'

        client.tear_down_task(task_type=task_name, task_idx=cur_idx)
        # client.reset(go_home=True)

        # Log end + duration
        dt = time.perf_counter() - t0
        wall_end = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log.info(f"END   {task_name}#{cur_idx} at {wall_end} "
                 f"duration={dt:.2f}s steps={steps} score={task_score}")
        return agent_successful, task_score, msg

    except Exception as e:
        dt = time.perf_counter() - t0
        wall_end = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log.info(f"FAIL  {task_name}#{cur_idx} at {wall_end} duration={dt:.2f}s "
                 f"error={e}")
        return False, 0.0, f"Exception during task {task_name}[{cur_idx}]: {e}"


def worker_run(env_url: str,
               tasks: List[Tuple[str, int]],
               model_name: str,
               max_steps: int,
               temperature: float,
               output_dir: str,
               save_img_dir: str,
               base_url: str,
               suite_reinit_args: Tuple[int, int, str]) -> Dict[str, Any]:
    """
    Runs a shard of (task_name, idx) on a single environment URL.
    Returns a summary dict.
    """
    # Per-worker file logger (with msec timestamps)
    file_logger = logging.getLogger(f"worker:{env_url}")
    file_logger.setLevel(logging.INFO)
    os.makedirs(output_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(output_dir, f"worker_{sanitize(env_url)}.log"))
    fh.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    # Avoid adding multiple handlers if the worker restarts
    if not file_logger.handlers:
        file_logger.addHandler(fh)

    worker_start = time.perf_counter()
    client = AndroidEnvClient(env_url)
    file_logger.info(f"Worker starting on {env_url} with {len(tasks)} tasks.")

    if not wait_until_healthy(client, max_wait_s=300, interval_s=1.0):
        file_logger.info("Environment never became healthy; aborting shard.")
        return {"env_url": env_url, "completed": 0, "success": 0, "failed": len(tasks)}

    # Reset & reinitialize suite for this environment to ensure a clean state
    client.reset(go_home=True)
    n_task_combinations, seed, task_family = suite_reinit_args
    re = client.reinitialize_suite(
        n_task_combinations=n_task_combinations, seed=seed, task_family=task_family
    )
    file_logger.info(f"Reinitialized suite: {re}")

    agent = build_agent(client, model_name, temperature=temperature, file_logger=file_logger, save_img_dir=save_img_dir, base_url=base_url)

    completed = 0
    success = 0
    for tname, true_task_id, tidx in tasks:
        ok, score, msg = eval_one_task(client, agent, tname, true_task_id, tidx, max_steps, file_logger)
        file_logger.info(f"[{tname}#{tidx}] {msg}")
        completed += 1
        success += int(ok)

        # Optional: brief pause between tasks to let emulator settle
        time.sleep(0.0)

    # # Best-effort close
    # try:
    #     client.close()
    # except Exception as e:
    #     file_logger.info(f"Close error: {e}")

    total_sec = time.perf_counter() - worker_start
    file_logger.info(f"Worker finished in {total_sec:.2f}s "
                     f"(completed={completed}, success={success}, failed={completed - success})")
    return {"env_url": env_url, "completed": completed, "success": success, "failed": completed - success}


def sanitize(url: str) -> str:
    return url.replace("://", "_").replace("/", "_").replace(":", "_")


def expand_suite(client: AndroidEnvClient, suite_max_index: int) -> List[Tuple[str, int]]:
    """
    Builds the global list of (task_name, idx) pairs by querying the suite.
    """
    task_list = client.get_suite_task_list(max_index=suite_max_index)
    pairs: List[Tuple[str, int]] = []
    for k, tname in enumerate(task_list):
        n = client.get_suite_task_length(task_type=tname)
        for i in range(n):
            pairs.append((tname, k, i))
    return pairs


def main():
    args = get_args()
    os.makedirs(args.output_path, exist_ok=True)

    env_urls = [u.strip() for u in args.env_urls.split(",") if u.strip()]
    if args.num_workers < 1:
        args.num_workers = 1
    if args.num_workers > len(env_urls):
        logger.info(f"--num_workers ({args.num_workers}) > env_urls ({len(env_urls)}). "
                    f"Using {len(env_urls)} workers.")
        args.num_workers = len(env_urls)

    # Query suite only once (using the first env) to get the full set of tasks
    probe_client = AndroidEnvClient(env_urls[0])
    if not wait_until_healthy(probe_client, max_wait_s=300, interval_s=1.0):
        raise RuntimeError(f"Probe environment {env_urls[0]} did not become healthy.")

    # (Optional) Reset probe and reinitialize so the listing is fresh
    probe_client.reset(go_home=True)
    _ = probe_client.reinitialize_suite(
        n_task_combinations=args.reinit_n_task_combinations,
        seed=args.reinit_seed,
        task_family=args.reinit_task_family,
    )

    all_pairs = expand_suite(probe_client, suite_max_index=args.suite_max_index)
    logger.info(f"Discovered {len(all_pairs)} (task_name, idx) pairs.")

    # Honor --task_index and --max_tasks like original
    if args.task_index >= 0:
        all_pairs = all_pairs[args.task_index:]
        logger.info(f"Starting from task_index={args.task_index}; remaining {len(all_pairs)} tasks.")
    if args.max_tasks > 0:
        all_pairs = all_pairs[:args.max_tasks]
        logger.info(f"Restricting to max_tasks={args.max_tasks}; using {len(all_pairs)} tasks.")

    if not all_pairs:
        logger.info("No tasks to run. Exiting.")
        return

    # Shard tasks round-robin across workers (bind workers 1:1 to env_urls[:num_workers])
    num_workers = args.num_workers
    worker_envs = env_urls[:num_workers]
    shards: List[List[Tuple[str, int]]] = [[] for _ in range(num_workers)]
    for i, pair in enumerate(all_pairs):
        shards[i % num_workers].append(pair)

    logger.info("Launching workers:")
    for w, (env, shard) in enumerate(zip(worker_envs, shards)):
        logger.info(f"  Worker {w}: {env} -> {len(shard)} tasks")

    
    # use only one base_url for all workers
    base_urls = [args.base_url]
    base_urls_for_workers = [base_urls[0] for _ in range(num_workers)] 

    # Fire the workers
    results = []
    suite_reinit_args = (args.reinit_n_task_combinations, args.reinit_seed, args.reinit_task_family)
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futs = [
            ex.submit(
                worker_run,
                env_url=env,
                tasks=shard,
                model_name=args.model,
                temperature=args.temperature,
                max_steps=args.max_steps,
                output_dir=args.output_path,
                save_img_dir=args.save_img_dir,
                suite_reinit_args=suite_reinit_args,
                base_url=base_urls_for_workers[i],
            )
            for i, (env, shard) in enumerate(zip(worker_envs, shards))
        ]
        for f in as_completed(futs):
            results.append(f.result())

    # Print a concise summary with totals
    total_completed = sum(r["completed"] for r in results)
    total_success = sum(r["success"] for r in results)
    total_failed = sum(r["failed"] for r in results)

    logger.info("========== SUMMARY ==========")
    for r in results:
        logger.info(f"{r['env_url']}: completed={r['completed']} success={r['success']} failed={r['failed']}")
    logger.info(f"TOTAL: completed={total_completed} success={total_success} failed={total_failed}")


if __name__ == "__main__":
    main()
