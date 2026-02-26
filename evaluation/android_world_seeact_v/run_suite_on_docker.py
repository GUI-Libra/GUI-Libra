# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-process evaluation client for Android World over HTTP.

Prerequisites:

# Build the Docker image for the Android environment server from the root
repository directory
# docker build -t android_world:latest .

# Run the Docker container
# docker run --privileged -p 5000:5000 -it android_world:latest

After running the server, you can use the client to interact with the
environment. You'll need to implement your agent logic to interact with the
environment.
"""
import os
import json
import logging
import time
from typing import Any
import argparse

from android_world.agents import infer
from android_world.env import json_action
from android_world.agents import (
    seeact_v,
    seeact_v_gpt_noselfsummary,
    seeact_v_guilibra,
    seeact_v_local_docker,
    seeact_v_aguvis,
    seeact_v_uitars,
    seeact_v_glm,
    seeact_v_qwen,
    seeact_v_qwen_step_summary,
)
import numpy as np
import pydantic
import requests

logger = logging.getLogger()
logger.setLevel(logging.INFO)

Params = dict[str, int | str]


def get_args():
    parser = argparse.ArgumentParser(description="Android World Agent Config")

    parser.add_argument(
        '--task_index',
        type=int,
        default=-1,
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='./test',
        help='The path to save results to.',
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=15,
        help='Maximum number of steps to run the agent.',
    )
    parser.add_argument(
        '--agent_name',
        type=str,
        default='seeact_v',
        help='Agent name.',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='The model to use for the agent.',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Temperature for the LLM.',
    )
    parser.add_argument(
        '--base_url',
        type=str,
        default='http://localhost:8000/v1',
        help='Base URL for LLM API.',
    )
    parser.add_argument(
        '--env_url',
        type=str,
        default='http://localhost:23333/emu1',
        help='Base URL of the Android emulator gateway.',
    )

    return parser.parse_args()


class Response(pydantic.BaseModel):
    status: str
    message: str


class AndroidEnvClient:
    """Client for interacting with the Android environment server."""

    def __init__(self, base_url: str = "http://localhost:23333/emu1"):
        logger.info(
            "Setting up Android environment using Docker - Initial setup may take"
            " 5-10 minutes. Please wait..."
        )
        self.base_url = base_url

    def reset(self, go_home: bool) -> Response:
        """Resets the environment."""
        response = requests.post(
            f"{self.base_url}/reset", params={"go_home": go_home}
        )
        response.raise_for_status()
        return Response(**response.json())

    def get_screenshot(
        self, wait_to_stabilize: bool = False
    ) -> np.ndarray[Any, Any]:
        """Gets the current screenshot of the environment."""
        response = requests.get(
            f"{self.base_url}/screenshot",
            params={"wait_to_stabilize": wait_to_stabilize},
        )
        response.raise_for_status()
        image = response.json()
        return np.array(image["pixels"])

    def execute_action(
        self,
        action: json_action.JSONAction,
    ) -> Response:
        """Executes an action in the environment."""
        response = requests.post(
            f"{self.base_url}/execute_action",
            json=json.loads(action.json_str()),
        )
        response.raise_for_status()
        return Response(**response.json())

    def get_suite_task_list(self, max_index: int) -> list[str]:
        """Gets the list of tasks in the suite."""
        response = requests.get(
            f"{self.base_url}/suite/task_list", params={"max_index": max_index}
        )
        response.raise_for_status()
        return response.json()["task_list"]

    def get_suite_task_length(self, task_type: str) -> int:
        """Gets the length of the suite of tasks."""
        response = requests.get(
            f"{self.base_url}/suite/task_length", params={"task_type": task_type}
        )
        response.raise_for_status()
        return response.json()["length"]

    def reinitialize_suite(
        self,
        n_task_combinations: int = 1,
        seed: int = 42,
        task_family: str = "android_world",
    ) -> Response:
        """Reinitializes the suite of tasks."""
        response = requests.get(
            f"{self.base_url}/suite/reinitialize",
            params={
                "n_task_combinations": n_task_combinations,
                "seed": seed,
                "task_family": task_family,
            },
        )
        response.raise_for_status()
        return Response(**response.json())

    def initialize_task(self, task_type: str, task_idx: int) -> Response:
        """Initializes the task in the environment."""
        params: Params = {"task_type": task_type, "task_idx": task_idx}
        response = requests.post(f"{self.base_url}/task/initialize", params=params)
        response.raise_for_status()
        return Response(**response.json())

    def tear_down_task(self, task_type: str, task_idx: int) -> Response:
        """Tears down the task in the environment."""
        params: Params = {"task_type": task_type, "task_idx": task_idx}
        response = requests.post(f"{self.base_url}/task/tear_down", params=params)
        response.raise_for_status()
        return Response(**response.json())

    def get_task_score(self, task_type: str, task_idx: int) -> float:
        """Gets the score of the current task."""
        params: Params = {"task_type": task_type, "task_idx": task_idx}
        response = requests.get(f"{self.base_url}/task/score", params=params)
        response.raise_for_status()
        return response.json()["score"]

    def get_task_goal(self, task_type: str, task_idx: int) -> str:
        """Gets the goal of the current task."""
        params: Params = {"task_type": task_type, "task_idx": task_idx}
        response = requests.get(f"{self.base_url}/task/goal", params=params)
        response.raise_for_status()
        return response.json()["goal"]

    def get_task_template(self, task_type: str, task_idx: int) -> str:
        """Gets the template of the current task."""
        params: Params = {"task_type": task_type, "task_idx": task_idx}
        response = requests.get(f"{self.base_url}/task/template", params=params)
        response.raise_for_status()
        return response.json()["template"]

    def close(self) -> None:
        """Closes the environment."""
        response = requests.post(f"{self.base_url}/close")
        response.raise_for_status()

    def health(self) -> bool:
        """Checks the health of the environment."""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Environment is not healthy: {e}")
            return False
        return True


def build_agent(client, model_name, temperature, base_url):
    """Build the appropriate agent based on model name."""
    if 'ray2333' in model_name.lower() or 'gui-libra' in model_name.lower():
        return seeact_v_guilibra.SeeAct_V(
            client,
            infer.Gpt4Wrapper(model_name, temperature=temperature, base_url=base_url),
        )
    elif 'qwen' in model_name.lower():
        if 'step-summary' in model_name.lower():
            model_name = model_name.replace('step-summary', '')
            return seeact_v_qwen_step_summary.SeeAct_V(
                client,
                infer.Gpt4Wrapper(model_name, temperature=temperature, base_url=base_url),
            )
        else:
            return seeact_v_qwen.SeeAct_V(
                client,
                infer.Gpt4Wrapper(model_name, temperature=temperature, base_url=base_url),
            )
    elif any(k in model_name.lower() for k in ("gpt", "4o", "4.1", "o4")):
        if 'gpt-5' in model_name.lower() or 'o4' in model_name.lower():
            temperature = 1.0
        return seeact_v_gpt_noselfsummary.SeeAct_V(
            client,
            infer.Gpt4Wrapper(model_name, temperature=temperature, base_url=base_url),
            grounding_model_name="osunlp/UGround-V1-7B",
        )
    elif 'glm' in model_name.lower():
        return seeact_v_glm.SeeAct_V(
            client,
            infer.Gpt4Wrapper(model_name, temperature=temperature, base_url=base_url),
            grounding_model_name="osunlp/UGround-V1-7B",
        )
    elif 'aguvis' in model_name.lower():
        return seeact_v_aguvis.SeeAct_V(
            client,
            infer.Gpt4Wrapper(model_name, temperature=temperature, base_url=base_url),
            grounding_model_name="osunlp/UGround-V1-7B",
        )
    elif 'tars' in model_name.lower() and 'bytedance' in model_name.lower():
        return seeact_v_uitars.SeeAct_V(
            client,
            infer.Gpt4Wrapper(model_name, temperature=temperature, base_url=base_url),
            grounding_model_name="osunlp/UGround-V1-7B",
        )
    else:
        return seeact_v_local_docker.SeeAct_V(
            client,
            infer.Gpt4Wrapper(model_name, temperature=temperature, base_url=base_url),
            grounding_model_name="osunlp/UGround-V1-7B",
        )


if __name__ == "__main__":
    args = get_args()
    print("Initializing Android environment client...")
    os.makedirs(args.output_path, exist_ok=True)

    client = AndroidEnvClient(base_url=args.env_url)

    while True:
        if not client.health():
            print("Environment is not healthy, waiting for 1 second...")
            time.sleep(1)
        else:
            break

    res = client.reset(go_home=True)
    print(f"reset response: {res}")

    task_list = client.get_suite_task_list(max_index=-1)
    if args.task_index >= 0:
        task_list = task_list[args.task_index:]
        print(f"Running tasks from index {args.task_index}: {task_list}")
    else:
        print(task_list)

    res = client.reinitialize_suite()
    print(f"reinitialize_suite response: {res}")

    agent = build_agent(client, args.model, args.temperature, args.base_url)

    for task_name in task_list:
        num_tasks = client.get_suite_task_length(task_type=task_name)
        print(f"num_tasks: {num_tasks}")

        for cur_idx in range(num_tasks):
            agent.reset(task_id=cur_idx, repeat_id=cur_idx)
            task_template = client.get_task_template(
                task_type=task_name, task_idx=cur_idx
            )
            if not ('{' in task_template and '}' in task_template) and cur_idx > 0:
                break

            task_goal = client.get_task_goal(task_type=task_name, task_idx=cur_idx)
            print(f"task_goal: {task_goal}")

            try:
                res = client.initialize_task(task_type=task_name, task_idx=cur_idx)
                print(f"Goal: {task_goal}")

                is_done = False
                for _ in range(args.max_steps):
                    response = agent.step(task_goal)
                    if response.done:
                        is_done = True
                        break

                task_score = client.get_task_score(
                    task_type=task_name, task_idx=cur_idx
                )
                print(f"task_score: {task_score}")
                agent_successful = is_done and task_score == 1
                print(
                    f'{"Task Successful ✅" if agent_successful else "Task Failed ❌"};'
                    f' {task_goal}'
                )

                res = client.tear_down_task(task_type=task_name, task_idx=cur_idx)

            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"Error initializing task {task_name} {cur_idx}: {e}")
                print("Continuing to next task...")
                continue

            res = client.reset(go_home=True)

    client.close()
