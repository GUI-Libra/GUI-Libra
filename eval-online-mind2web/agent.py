from syn.base_explore import Explorer, PlaywrightTimeoutError, step_timeout
from syn.data import (
    StateInfo, Action, ActionType, HighLevelTask, LowLevelTask, LowTaskStatus,
)
from syn.args import AgentConfig, APIProvider, is_qwen25_model
from syn.prompts import (
    build_guipivot_messages,
    build_guipivot_grounding_messages,
    build_uground_messages,
    parse_grounding_response,
    parse_plan_guipivot,
    parse_guipivot_json,
    convert_guipivot_to_exec_actions,
    guipivot_action_to_mind2web_str,
    _normalize_guipivot_action_type,
)
from syn.tools import (
    tools_ndarray_to_base64_image_raw,
    tools_get_time,
    tools_elapsed_time_print,
    tools_jsonl_save,
    tools_jsonl_load,
    tools_serialize_dataclass,
)
from syn.gpt import GPTClient
from syn.mind2web import Mind2WebResultWriter

import json
import time
import os
import random
import numpy as np
from loguru import logger
from simpleArgParser import parse_args
from tqdm import tqdm


class Agent(Explorer):
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.config: AgentConfig = config
        self.tasks_done_buffer: list[HighLevelTask] = []
        self.tasks_todo: list[dict] = []
        self.tasks_done_unique: dict[str, str] = {}
        self.load()
        os.makedirs(self.config.output, exist_ok=True)
        self.mind2web_writer = Mind2WebResultWriter(self.config.output)
        self._current_mind2web_task_id: str | None = None

        self._grounding_client: GPTClient | None = None
        if self.config.external_grounding_port:
            self._grounding_client = GPTClient(
                provider=APIProvider.openai,
                api_key="token-abc123",
                base_url=f"http://localhost:{self.config.external_grounding_port}/v1",
            )
            logger.info(f"Initialized grounding client on port {self.config.external_grounding_port}")

    def save(self):
        super().save()
        tools_jsonl_save(self.tasks_todo, f"{self.config.output}/tasks_todo.jsonl")
        tools_jsonl_save(tools_serialize_dataclass(self.tasks_done_buffer), f"{self.config.output}/tasks_done.jsonl", append=True)
        self.tasks_done_buffer = []
        with open(f"{self.config.output}/tasks_done_unique.json", 'w') as f:
            json.dump(self.tasks_done_unique, f, indent=4)

    def load(self):
        super().load()
        if os.path.exists(path := f"{self.config.output}/tasks_todo.jsonl"):
            self.tasks_todo = tools_jsonl_load(path)
            logger.info(f"Loaded {len(self.tasks_todo)} tasks from {path}, skipping input={self.config.tasks_path}")
        else:
            self.tasks_todo = tools_jsonl_load(self.config.tasks_path)
            if self.config.limit is not None and self.config.limit > 0:
                self.tasks_todo = self.tasks_todo[:self.config.limit]
            logger.info(f"Loaded {len(self.tasks_todo)} tasks from {self.config.tasks_path}")
        random.shuffle(self.tasks_todo)

        if os.path.exists(path := f"{self.config.output}/tasks_done_unique.json"):
            self.tasks_done_unique = json.load(open(path, 'r'))
            logger.info(f"Loaded {len(self.tasks_done_unique)} done tasks from {path}")

        self._task_crash_counts: dict[str, int] = {}
        crash_counts_path = f"{self.config.output}/task_crash_counts.json"
        if os.path.exists(crash_counts_path):
            try:
                self._task_crash_counts = json.load(open(crash_counts_path, 'r'))
            except Exception:
                pass

        crash_marker_path = f"{self.config.output}/current_task_marker.json"
        if os.path.exists(crash_marker_path):
            try:
                marker = json.load(open(crash_marker_path, 'r'))
                crashed_task_id = marker.get('task_unique_id', '')
                if crashed_task_id:
                    prev_count = self._task_crash_counts.get(crashed_task_id, 0)
                    self._task_crash_counts[crashed_task_id] = prev_count + 1
                    logger.warning(f"Detected crash on task '{crashed_task_id}' (crash count: {self._task_crash_counts[crashed_task_id]})")
                    with open(crash_counts_path, 'w') as f:
                        json.dump(self._task_crash_counts, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to read crash marker: {e}")
            try:
                os.remove(crash_marker_path)
            except OSError:
                pass

    def _call_grounding(self, description: str, base64_image: str) -> tuple[tuple[float, float], str]:
        """Call UGround grounding model. Returns ((gx, gy), raw_response_text)."""
        grounding_messages = build_uground_messages(description, base64_image)
        grounding_response = self._grounding_client.request(
            messages=grounding_messages,
            model="osunlp/UGround-V1-7B",
            temperature=0,
            max_completion_tokens=128,
            json_mode=False,
            stat_token_usage=False,
        )
        grounding_text = grounding_response.message.content
        logger.info(f"Grounding response for '{description}': {grounding_text}")
        gx, gy = parse_grounding_response(grounding_text)
        return (gx, gy), grounding_text


    def _cot_step(self, task: str, current_state: StateInfo, previous_traj: list[LowLevelTask]) -> LowLevelTask:
        failed_low_level_task = LowLevelTask(
            task="failed during cot_step",
            action=Action(element=None, action_type=ActionType.STOP, value="error during cot_step"),
            curr_state=current_state,
            task_status=LowTaskStatus.IN_PROGRESS,
        )

        error_msg = None
        response_text = ""
        grounding_info = None  # will hold grounding debug info if used

        screenshot = current_state.raw_state.screenshot
        if screenshot is None:
            error_msg = "GUIPivot requires screenshot but none available"
            failed_low_level_task.action.value = error_msg
            logger.error(error_msg)
            return failed_low_level_task

        base64_image, img_size = tools_ndarray_to_base64_image_raw(screenshot)
        image_width, image_height = img_size

        history_last_k = self.config.history_last_k or 15
        previous_operations = []
        for low_task in previous_traj[-history_last_k:]:
            operation = low_task.task or ""
            if not operation and low_task.action:
                operation = str(low_task.action)
            previous_operations.append(operation)

        use_grounding = self._grounding_client is not None

        if use_grounding:
            messages = build_guipivot_grounding_messages(
                task=task,
                base64_image=base64_image,
                image_width=image_width,
                image_height=image_height,
                previous_actions=previous_operations,
                last_k=history_last_k,
            )
        else:
            messages = build_guipivot_messages(
                task=task,
                base64_image=base64_image,
                image_width=image_width,
                image_height=image_height,
                previous_actions=previous_operations,
                last_k=history_last_k,
            )

        try:
            response = self.gpt_client.request(
                messages=messages,
                json_mode=False,
                **self.config.gpt.__dict__,
            )
            response_text = response.message.content

            thought, operation, action_json_str = parse_plan_guipivot(response_text)

            if not action_json_str:
                error_msg = f"Failed to extract <answer> from response: {response_text}"
                failed_low_level_task.action.value = error_msg
                logger.error(error_msg)
                return failed_low_level_task

            try:
                data = parse_guipivot_json(action_json_str)
            except Exception as e:
                error_msg = f"Failed to parse JSON: {e}\naction_json_str={action_json_str}"
                failed_low_level_task.action.value = error_msg
                logger.error(error_msg)
                return failed_low_level_task

            if "action_description" in data:
                operation = data["action_description"]

            # --- Grounding: inject coordinates from UGround ---
            if use_grounding:
                normalized_type = _normalize_guipivot_action_type(data.get("action_type", ""))
                if normalized_type in ("Click", "Select", "Write"):
                    grounding_desc = data.get("action_target") or data.get("action_description", "")
                    if grounding_desc and grounding_desc.lower() not in ("none", "null", ""):
                        try:
                            (gx, gy), grounding_raw = self._call_grounding(grounding_desc, base64_image)
                            data["point_2d"] = [gx, gy]
                            grounding_info = {
                                "description": grounding_desc,
                                "raw_response": grounding_raw,
                                "coordinates": [gx, gy],
                                "scaled_coordinates": [
                                    (gx / 1000) * image_width,
                                    (gy / 1000) * image_height,
                                ],
                            }
                            logger.info(f"Grounding injected point_2d={data['point_2d']} for '{grounding_desc}'")
                        except Exception as e:
                            logger.error(f"Grounding failed for '{grounding_desc}': {e}")
                            grounding_info = {"description": grounding_desc, "error": str(e)}

            use_smart_resize = is_qwen25_model(self.config.gpt.model)

            exec_actions = convert_guipivot_to_exec_actions(
                data=data,
                screen_width=image_width,
                screen_height=image_height,
                use_smart_resize=use_smart_resize if not use_grounding else False,
            )

            if not exec_actions:
                error_msg = f"No executable actions from response: {data}"
                failed_low_level_task.action.value = error_msg
                logger.error(error_msg)
                return failed_low_level_task

            primary_action = exec_actions[0]
            action_name = primary_action.get("name", "")
            params = primary_action.get("parameters", {})

            action_type_mapping = {
                "click": ActionType.CLICK,
                "write": ActionType.TYPE,
                "press": ActionType.PRESS,
                "swipe": ActionType.SCROLL,
                "back": ActionType.GO_BACK,
                "response": ActionType.NONE,
                "terminate": ActionType.NONE,
                "wait": ActionType.PRESS,
            }
            action_type = action_type_mapping.get(action_name, ActionType.PRESS)

            coordinates = None
            value = ""

            if action_name == "click":
                x, y = params.get("x"), params.get("y")
                if x is not None and y is not None:
                    coordinates = (x, y)
            elif action_name == "write":
                value = params.get("message", "")
                if len(exec_actions) > 1 and exec_actions[0].get("name") == "click":
                    click_params = exec_actions[0].get("parameters", {})
                    x, y = click_params.get("x"), click_params.get("y")
                    if x is not None and y is not None:
                        coordinates = (x, y)
            elif action_name == "press":
                value = params.get("keys", "")
            elif action_name == "swipe":
                swipe_dir = params.get("direction", "up")
                direction_mapping = {"up": "down", "down": "up", "left": "right", "right": "left"}
                value = direction_mapping.get(swipe_dir, swipe_dir)
            elif action_name in ["response", "terminate"]:
                value = params.get("answer", "") or params.get("info", "") or "Task Completed"
            elif action_name == "wait":
                seconds = params.get("seconds", 1)
                value = f"wait {seconds}s"
            else:
                value = f"noop ({action_name})"

            if Action._is_required_element(action_type) and coordinates is None:
                logger.warning(f"action_type={action_type} requires coordinates but none available. Downgrading to PRESS.")
                action_type = ActionType.PRESS
                if not value or not value.strip():
                    value = f"noop (missing coords for {action_name})"

            current_state.summary = thought

            action = Action(
                action_type=action_type,
                element=None,
                value=value,
                coordinates=coordinates,
            )
            action.exec_actions = exec_actions
            action.raw_response = response_text
            action.raw_input_messages = messages
            if grounding_info is not None:
                action.grounding_info = grounding_info

            low_level_instruction = operation or guipivot_action_to_mind2web_str(primary_action)

            low_level_task = LowLevelTask(
                task=low_level_instruction,
                curr_state=current_state,
                action=action,
                task_status=LowTaskStatus.IN_PROGRESS,
                reasoning=thought,
            )

            logger.debug(f"next_action: {low_level_instruction}, action_type={action_type}, coordinates={coordinates}\nresponse_text={response_text}")

        except Exception as e:
            error_msg = f"error during cot_step for task={task} with error={e}\nresponse={response_text}"

        if isinstance(error_msg, str):
            failed_low_level_task.action.value = error_msg
            logger.error(error_msg)
            return failed_low_level_task

        return low_level_task

    def _compose_last_action_summary(self, high_level_task: HighLevelTask) -> str:
        last_low_task = high_level_task.trajectories[-1]
        summary = None
        if last_low_task.action.action_type in {ActionType.NONE, ActionType.STOP}:
            summary = last_low_task.action.value
        if summary is None or str(summary).strip() == "":
            if last_low_task.state_after and isinstance(last_low_task.state_after.summary, str):
                summary = last_low_task.state_after.summary
            elif last_low_task.reasoning:
                summary = last_low_task.reasoning
            elif last_low_task.task:
                summary = last_low_task.task
            else:
                summary = "No summary available"
        return str(summary)

    def _start_mind2web_logging(self, task_id: str, task: str, start_url: str, initial_state: StateInfo, task_dict: dict):
        self._current_mind2web_task_id = task_id
        self.mind2web_writer.start_task(
            task_id=task_id,
            task=task,
            start_url=start_url,
            input_image_paths=task_dict.get("input_image_paths"),
        )
        self.mind2web_writer.log_initial_state(initial_state)

    def _log_mind2web_step(self, low_level_task: LowLevelTask, state_after: StateInfo | None):
        if self._current_mind2web_task_id is None:
            return
        self.mind2web_writer.log_step(low_level_task, state_after)

    def _finalize_mind2web_task(self, task_status: dict, high_level_task: HighLevelTask):
        if self._current_mind2web_task_id is None:
            return
        final_summary = self._compose_last_action_summary(high_level_task)
        self.mind2web_writer.finalize(task_status.get('end_reason', 'unknown'), final_summary)
        self._current_mind2web_task_id = None

    def _stat_accuracy(self, execute_status: dict[str, str]) -> tuple[int, int, int, int]:
        complete_cnt = sum(1 for status in execute_status.values() if status['end_reason'] == 'completed')
        total_cnt = len(execute_status)
        return 0, 0, complete_cnt, total_cnt

    def run_episode(self):
        if len(self.tasks_todo) == 0:
            raise ValueError("No tasks provided for execution.")
        first_start_url = self.tasks_todo[0].get('start_url')
        if not isinstance(first_start_url, str) or len(first_start_url) == 0:
            raise ValueError("Tasks must include 'start_url'.")

        env, current_state = self._init_env_for_episode(first_start_url)
        task_exe_cnt = len(self.tasks_done_unique)

        for task_cnt in tqdm(range(len(self.tasks_todo)), desc=f'agent-{self.config.output}', initial=task_exe_cnt, total=len(self.tasks_todo)):
            task_dict = self.tasks_todo[task_cnt]
            task = task_dict['task']
            task_id = task_dict.get('task_id') or task_dict.get('id') or task
            start_url = task_dict.get('start_url', first_start_url)

            if not isinstance(start_url, str) or len(start_url) == 0:
                raise ValueError(f"Invalid start_url for task={task}")

            if task_id in self.tasks_done_unique:
                logger.info(f"Task '{task}' already done, skipping.")
                continue

            MAX_CRASHES_PER_TASK = 3
            crash_count = self._task_crash_counts.get(task_id, 0)
            if crash_count >= MAX_CRASHES_PER_TASK:
                logger.error(f"Task '{task}' crashed {crash_count} times. Marking as FAILED.")
                task_status = {'end_reason': f'crashed_{crash_count}_times', 'auto-eval': 'NA'}
                self.mind2web_writer.start_task(
                    task_id=str(task_id), task=task, start_url=start_url,
                    input_image_paths=task_dict.get("input_image_paths"),
                )
                self.mind2web_writer.finalize(task_status['end_reason'], f"Task crashed {crash_count} times.")
                self.tasks_done_unique[task_id] = task_status
                self.save()
                continue

            crash_marker_path = f"{self.config.output}/current_task_marker.json"
            try:
                with open(crash_marker_path, 'w') as f:
                    json.dump({'task_unique_id': task_id, 'task': task}, f)
                    f.flush()
                    os.fsync(f.fileno())
            except Exception as e:
                logger.warning(f"Failed to write crash marker: {e}")

            try:
                if not self._check_browser_alive(env):
                    logger.warning(f"Browser appears dead. Recovering...")
                    env, current_state = self._recover_browser(env, start_url)
                else:
                    observation, info = self._reset_env(env, start_url=start_url)
                    observation_metadata = info['observation_metadata']
                    current_state = self._get_env_state(env, obs=observation, observation_metadata=observation_metadata)
            except Exception as reset_err:
                logger.error(f"Failed to reset env: {reset_err}. Recovering browser...")
                try:
                    env, current_state = self._recover_browser(env, start_url)
                except Exception as recovery_err:
                    logger.error(f"Browser recovery failed: {recovery_err}. Skipping task.")
                    continue

            logger.info(f"--- Start Executing {task_exe_cnt}/{len(self.tasks_todo)} ---\nTask={task}\nstart_url={start_url}")
            logger.info(f"total gpt usage:\n{self.gpt_client.token_usage}")

            try:
                current_state = self._reset_all_tabs_and_open_seed_url(env, start_url)
            except Exception as tabs_err:
                logger.error(f"Failed to reset tabs: {tabs_err}. Recovering browser...")
                try:
                    env, current_state = self._recover_browser(env, start_url)
                except Exception as recovery_err:
                    logger.error(f"Browser recovery failed: {recovery_err}. Skipping task.")
                    continue

            self._start_mind2web_logging(str(task_id), task, start_url, current_state, task_dict)

            high_level_task = HighLevelTask(task=task, start_url=start_url, trajectories=[])

            step_idx = 0
            failed_attempt = 0
            browser_dead = False

            while step_idx < self.config.max_steps and failed_attempt < self.config.failed_retry:
                logger.info(f"Step {step_idx + 1}/max={self.config.max_steps} for task={task}, failed_attempt={failed_attempt}/{self.config.failed_retry}")

                try:
                    next_low_level_task = self._cot_step(high_level_task.task, current_state, high_level_task.trajectories)
                except PlaywrightTimeoutError as e:
                    logger.error(f"Browser timeout during COT step: {e}")
                    try:
                        env, current_state = self._recover_browser(env, start_url)
                    except Exception:
                        pass
                    browser_dead = True
                    break

                high_level_task.trajectories.append(next_low_level_task)
                state_for_logging: StateInfo | None = None
                should_break = False

                if next_low_level_task.action.action_type is ActionType.NONE:
                    next_low_level_task.task_status = LowTaskStatus.END
                    logger.info(f"Task completed: {high_level_task.task}")
                    state_for_logging = current_state
                    should_break = True

                elif next_low_level_task.action.action_type is ActionType.STOP:
                    next_low_level_task.task_status = LowTaskStatus.NOTACHIEVEABLE
                    failed_attempt += 1
                    if failed_attempt < self.config.failed_retry:
                        low_task = high_level_task.trajectories[-1]
                        low_task.task_status = LowTaskStatus.IN_PROGRESS
                        low_task.action.action_type = ActionType.REFLECT
                        low_task.action.value = f"**Failed Analysis**: {low_task.action.value}.\n**Reflection**: Try DIFFERENT approaches."
                        high_level_task.trajectories[-1] = low_task
                    state_for_logging = current_state

                else:
                    try:
                        with step_timeout(self.STEP_TIMEOUT if hasattr(self, 'STEP_TIMEOUT') else 120, f"step {step_idx}", env=env):
                            next_state = self._execute_single_low_level_task(next_low_level_task, env, curr_state=current_state)
                        next_low_level_task.state_after = next_state
                        current_state = next_state
                        state_for_logging = current_state
                    except PlaywrightTimeoutError as e:
                        logger.error(f"Browser timeout during action: {e}")
                        try:
                            env, current_state = self._recover_browser(env, start_url)
                        except Exception:
                            pass
                        browser_dead = True
                        break
                    except Exception as e:
                        error_msg = str(e).lower()
                        if any(kw in error_msg for kw in ['connection', 'closed', 'broken pipe', 'target page', 'browser has been closed', 'protocol error']):
                            logger.error(f"Browser connection lost: {e}")
                            try:
                                env, current_state = self._recover_browser(env, start_url)
                            except Exception:
                                pass
                            browser_dead = True
                            break
                        else:
                            raise

                self._log_mind2web_step(next_low_level_task, state_for_logging)

                if should_break:
                    break
                step_idx += 1

            task_exe_cnt += 1
            self.gpt_client.token_usage.iteration_count += 1

            task_status = {
                'steps': len(high_level_task.trajectories),
                'max_steps': self.config.max_steps,
                'end_reason': 'unknown',
                'auto-eval': 'NA',
            }

            if browser_dead:
                s = "browser_timeout_recovery"
            elif step_idx >= self.config.max_steps:
                s = "exceeded_max_steps"
            elif high_level_task.trajectories and high_level_task.trajectories[-1].task_status == LowTaskStatus.END:
                s = "completed"
            elif high_level_task.trajectories and high_level_task.trajectories[-1].task_status == LowTaskStatus.NOTACHIEVEABLE:
                s = "not_achievable"
            else:
                s = "unknown"
            task_status['end_reason'] = s

            self._finalize_mind2web_task(task_status, high_level_task)
            self.tasks_done_unique[task_id] = task_status
            self.tasks_done_buffer.append(high_level_task)

            try:
                if os.path.exists(crash_marker_path):
                    os.remove(crash_marker_path)
            except OSError:
                pass

            logger.info(f"Task {task_id} done with {len(high_level_task.trajectories)} steps. Total done: {len(self.tasks_done_unique)}")
            _, _, complete_cnt, total_cnt = self._stat_accuracy(self.tasks_done_unique)
            logger.info(f"Complete rate: {complete_cnt}/{total_cnt}={complete_cnt / total_cnt if total_cnt > 0 else 0:.4f}")

            if len(self.tasks_done_buffer) > 0:
                self.save()

        env.close()
        logger.info(f"Episode finished. Done {len(self.tasks_done_unique)} tasks.")
        logger.info(f"Total GPT usage:\n{self.gpt_client.token_usage}")
        logger.info(f"Per iteration GPT usage:\n{self.gpt_client.token_usage.per_iteration_str()}")


if __name__ == "__main__":
    args: AgentConfig = parse_args(AgentConfig)
    start_time = tools_get_time()
    logger.info(f"Starting Agent with config\n{args}\nStart time: {start_time}")
    agent = Agent(args)
    agent.run_episode()
    logger.info(f"Agent done! Started at {start_time}, Elapsed: {tools_elapsed_time_print(start_time)}\n{args}")
