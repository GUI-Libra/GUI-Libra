import multiprocessing as mp
import os
import json
import math
import random
import copy
import time
from dataclasses import dataclass, field
from loguru import logger

from syn.args import AgentConfig
from syn.tools import (
    tools_get_time,
    tools_elapsed_time_print,
    tools_jsonl_save,
    tools_jsonl_load,
)
from agent import Agent
from simpleArgParser import parse_args
from tqdm import tqdm


@dataclass
class MultiAgentConfig(AgentConfig):
    num_processes: int = field(default=4, kw_only=True)

    def pre_process(self):
        super().pre_process()
        self.num_processes = max(1, self.num_processes)


def run_single_agent(idx: int, shared_configs: list[AgentConfig]):
    config = shared_configs[idx]
    if idx != 0:
        logger.remove()
    log_file = f"{config.output}/run.log"
    logger.add(
        log_file,
        format='<green>{time:YY-MM-DD HH:mm:ss.SS}</green> | <level>{level: <5}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>\n<level>{message}</level>',
        level='DEBUG', colorize=False, rotation=None,
    )
    logger.info(f"Running Agent={idx}/{len(shared_configs)-1} with config=\n{config}")
    agent = Agent(config)
    agent.run_episode()
    logger.info(f"Agent={idx}/{len(shared_configs)-1} finished with output={config.output}")


def run_monitoring_process(multi_config: MultiAgentConfig, shared_configs: list[AgentConfig], interval_minutes: int = 10):
    logger.remove()
    log_file = f"{multi_config.output}/run_monitor.log"
    logger.add(
        log_file,
        format='<green>{time:YY-MM-DD HH:mm:ss.SS}</green> | <level>{level: <5}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>\n<level>{message}</level>',
        level='DEBUG', colorize=False, rotation=None,
    )
    logger.info(f"Starting monitoring with {interval_minutes}min intervals")
    temp = MultiAgent(multi_config)
    temp.shared_configs = shared_configs
    interval_seconds = interval_minutes * 60

    while True:
        try:
            time.sleep(interval_seconds)
            temp.gather_results()
            temp.save()
            _, _, complete_cnt, total_cnt = temp._stat_accuracy(temp.tasks_done_unique)
            logger.info(
                f"Complete rate: {complete_cnt}/{total_cnt}={complete_cnt / total_cnt if total_cnt > 0 else 0:.4f}\n"
                f"Progress: {total_cnt}/{len(temp.tasks_todo)}={total_cnt / len(temp.tasks_todo) if len(temp.tasks_todo) > 0 else 0:.4f}"
            )
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            continue


class MultiAgent(Agent):
    def __init__(self, config: MultiAgentConfig):
        self.multi_config = config
        temp_params = copy.deepcopy(config.__dict__)
        del temp_params['num_processes']
        config_for_single = AgentConfig(**temp_params)
        super().__init__(config_for_single)
        del temp_params['output']
        self.shared_configs = [
            AgentConfig(**temp_params, output=f"{self.multi_config.output}/multiagent/{i}")
            for i in range(self.multi_config.num_processes)
        ]

    def gather_results(self):
        gpt_result = {'count': 0, 'call': 0, 'usage': {}}
        self.tasks_done_unique = {}
        self.tasks_todo = []
        self.tasks_done_buffer = []

        parent_dir = self.multi_config.output
        if os.path.exists(path := f"{parent_dir}/tasks_done_unique.json"):
            try:
                temp = json.load(open(path, 'r'))
                self.tasks_done_unique.update(temp)
                logger.info(f"Loaded {len(temp)} tasks_done_unique from parent dir")
            except Exception:
                pass

        multiagent_base = f"{parent_dir}/multiagent"
        all_dirs = []
        if os.path.isdir(multiagent_base):
            for entry in sorted(os.listdir(multiagent_base)):
                subdir = os.path.join(multiagent_base, entry)
                if os.path.isdir(subdir):
                    all_dirs.append(subdir)
        for config in self.shared_configs:
            if config.output not in all_dirs:
                all_dirs.append(config.output)

        seen_todo_ids = set()
        for subdir in all_dirs:
            if os.path.exists(path := f"{subdir}/gpt_client_token_usage.json"):
                temp = json.load(open(path, 'r'))
                gpt_result['count'] += temp.get('count', 0)
                gpt_result['call'] += temp.get('call', 0)
                for key, value in temp.get('usage', {}).items():
                    if key not in gpt_result['usage']:
                        gpt_result['usage'][key] = {}
                    for uk, uv in value.items():
                        if uk not in gpt_result['usage'][key]:
                            gpt_result['usage'][key][uk] = 0
                        gpt_result['usage'][key][uk] += uv

            if os.path.exists(path := f"{subdir}/tasks_done_unique.json"):
                temp = json.load(open(path, 'r'))
                self.tasks_done_unique.update(temp)

            if os.path.exists(path := f"{subdir}/tasks_todo.jsonl"):
                temp = tools_jsonl_load(path)
                for task_dict in temp:
                    tid = task_dict.get('task_id') or task_dict.get('id') or task_dict.get('task')
                    if tid not in seen_todo_ids:
                        seen_todo_ids.add(tid)
                        self.tasks_todo.append(task_dict)

        if gpt_result['count'] > 0:
            with open(path := f"{self.multi_config.output}/gpt_client_token_usage.json", 'w') as f:
                json.dump(gpt_result, f, indent=4)
            self.gpt_client.token_usage.load_from_json(path)

        if len(self.tasks_todo) == 0:
            self.tasks_todo = tools_jsonl_load(self.config.tasks_path)
            if self.config.limit is not None and self.config.limit > 0:
                self.tasks_todo = self.tasks_todo[:self.config.limit]
            logger.info(f"Loaded {len(self.tasks_todo)} tasks from {self.config.tasks_path}")
        else:
            logger.info(f"Loaded {len(self.tasks_todo)} tasks from {len(all_dirs)} subprocess dirs")

        logger.info(f"gather_results: {len(self.tasks_done_unique)} done, {len(self.tasks_todo)} todo")

    def distribute_tasks(self):
        random.shuffle(self.tasks_todo)
        tasks_per_process = math.ceil(len(self.tasks_todo) / self.multi_config.num_processes)
        for i in range(self.multi_config.num_processes):
            config = self.shared_configs[i]
            os.makedirs(config.output, exist_ok=True)
            start = i * tasks_per_process
            end = (i + 1) * tasks_per_process
            task_chunk = self.tasks_todo[start:end]
            if task_chunk:
                tools_jsonl_save(task_chunk, f"{config.output}/tasks_todo.jsonl")
            json.dump(self.tasks_done_unique, open(f"{config.output}/tasks_done_unique.json", 'w'), indent=4)

    def save(self):
        for idx, config in enumerate(self.shared_configs):
            if os.path.exists(f"{config.output}/run.log"):
                os.system(f"cp {config.output}/run.log {self.multi_config.output}/run_{idx}.log")
        super().save()

    def run_episode(self):
        import time as _time
        error_code = 0
        start_time = tools_get_time()
        logger.info(f"Starting MultiAgent with {self.multi_config.num_processes} processes")
        logger.info(f"Config: {self.multi_config}")

        self.gather_results()
        self.save()
        self.load()
        self.distribute_tasks()

        processes = []
        for process_id in range(self.multi_config.num_processes):
            p = mp.Process(target=run_single_agent, args=(process_id, self.shared_configs))
            p.start()
            processes.append(p)
            logger.info(f"Started process {process_id} with PID {p.pid}")

        monitoring = mp.Process(target=run_monitoring_process, args=(self.multi_config, self.shared_configs, 10))
        monitoring.start()

        MAX_PROCESS_WAIT_TIME = 3600 * 6
        MAX_RESTARTS = 100
        restart_counts = {i: 0 for i in range(self.multi_config.num_processes)}
        process_start_times = {i: _time.time() for i in range(self.multi_config.num_processes)}

        alive = list(enumerate(processes))
        while alive:
            still_alive = []
            for idx, p in alive:
                p.join(timeout=30)
                if p.is_alive():
                    elapsed = _time.time() - process_start_times[idx]
                    if elapsed > MAX_PROCESS_WAIT_TIME:
                        logger.warning(f"Process {idx} exceeded {MAX_PROCESS_WAIT_TIME}s, killing...")
                        p.kill()
                        p.join(timeout=10)
                        error_code = -9
                    else:
                        still_alive.append((idx, p))
                else:
                    logger.info(f"Process {idx} pid={p.pid} finished with exit code {p.exitcode}")
                    if p.exitcode != 0:
                        error_code = p.exitcode
                        if restart_counts[idx] < MAX_RESTARTS:
                            restart_counts[idx] += 1
                            logger.warning(f"Process {idx} died (exit={p.exitcode}). Restarting ({restart_counts[idx]}/{MAX_RESTARTS})...")
                            new_p = mp.Process(target=run_single_agent, args=(idx, self.shared_configs))
                            new_p.start()
                            processes[idx] = new_p
                            process_start_times[idx] = _time.time()
                            still_alive.append((idx, new_p))
                        else:
                            logger.error(f"Process {idx} exceeded max restarts.")
            alive = still_alive

        monitoring.kill()

        self.gather_results()
        self.save()
        _, _, complete_cnt, total_cnt = self._stat_accuracy(self.tasks_done_unique)
        logger.info(
            f"Complete rate: {complete_cnt}/{total_cnt}={complete_cnt / total_cnt if total_cnt > 0 else 0:.4f}\n"
            f"Progress: {total_cnt}/{len(self.tasks_todo)}={total_cnt / len(self.tasks_todo) if len(self.tasks_todo) > 0 else 0:.4f}"
        )
        logger.info(f"MultiAgent completed! Elapsed: {tools_elapsed_time_print(start_time)}")
        return error_code


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    args: MultiAgentConfig = parse_args(MultiAgentConfig)
    multiagent = MultiAgent(args)
    retry_cnt = 3
    error_code = 0
    while retry_cnt > 0:
        retry_cnt -= 1
        error_code = multiagent.run_episode()
        if error_code == 0:
            break
        logger.error(f"run_episode failed with error_code={error_code}, {retry_cnt} retries left")
    logger.info(f"MultiAgent finished with error_code={error_code}")
