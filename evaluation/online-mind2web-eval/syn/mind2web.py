import os
from dataclasses import dataclass, field
from typing import List, Optional
import json
import numpy as np
from PIL import Image
from syn.data import ActionType, LowLevelTask, StateInfo


def _save_png(image: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.fromarray(image)
    img.save(path, format="PNG")
    return path


def _json_dumps(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=4)

def format_action_for_mind2web(task: LowLevelTask) -> str:
    action = task.action
    action_type = action.action_type.name if action.action_type else "UNKNOWN"

    exec_actions = getattr(action, 'exec_actions', None)
    if exec_actions:
        parts = []
        for exec_action in exec_actions:
            name = exec_action.get("name", "")
            params = exec_action.get("parameters", {})

            if name == "click":
                x, y = params.get("x"), params.get("y")
                if x is not None and y is not None:
                    parts.append(f"CLICK at ({int(x)}, {int(y)})")
                else:
                    parts.append("CLICK")
            elif name == "write":
                message = params.get("message", "")
                parts.append(f"TYPE \"{message}\"")
            elif name == "press":
                keys = params.get("keys", "")
                presses = params.get("presses", 1)
                if presses > 1:
                    parts.append(f"PRESS {keys} x{presses}")
                else:
                    parts.append(f"PRESS {keys}")
            elif name == "swipe":
                direction = params.get("direction", "up")
                display_mapping = {"up": "down", "down": "up", "left": "right", "right": "left"}
                display_dir = display_mapping.get(direction, direction)
                parts.append(f"SCROLL {display_dir}")
            elif name == "back":
                parts.append("GO BACK")
            elif name == "wait":
                seconds = params.get("seconds", 3)
                parts.append(f"WAIT {seconds}s")
            elif name == "response":
                answer = params.get("answer", "")
                parts.append(f"ANSWER -> {answer}")
            elif name == "terminate":
                status = params.get("status", "success")
                info = params.get("info", "")
                parts.append(f"TERMINATE ({status}) -> {info}")

        if parts:
            return " -> ".join(parts)

    if action.coordinates and action.target_element is None:
        coord_str = f"({int(action.coordinates[0])}, {int(action.coordinates[1])})"
        match action.action_type:
            case ActionType.CLICK:
                return f"CLICK at {coord_str}"
            case ActionType.TYPE:
                value = action.value.strip() if action.value else ""
                return f"TYPE \"{value}\" at {coord_str}"
            case ActionType.HOVER:
                return f"HOVER at {coord_str}"
            case ActionType.SELECT:
                value = action.value.strip() if action.value else ""
                return f"SELECT \"{value}\" at {coord_str}"
            case ActionType.SCROLL:
                direction = action.value.strip() if isinstance(action.value, str) else "down"
                return f"SCROLL {direction}"
            case ActionType.PRESS:
                return f"PRESS {action.value.strip() if action.value else ''}"
            case ActionType.GO_BACK:
                return "GO BACK"
            case ActionType.NONE:
                return f"ANSWER -> {action.value}".strip()
            case _:
                return f"{action_type} at {coord_str}"

    target = action.target_element.accessibility_tree_content if action.target_element else ""
    match action.action_type:
        case ActionType.CLICK:
            return f"{target} -> CLICK"
        case ActionType.TYPE:
            return f"{target} -> TYPE \"{action.value.strip() if action.value else ''}\""
        case ActionType.HOVER:
            return f"{target} -> HOVER"
        case ActionType.SCROLL:
            direction = action.value.strip() if isinstance(action.value, str) else "down"
            return f"SCROLL {direction}"
        case ActionType.PRESS:
            return f"PRESS {action.value.strip() if action.value else ''}"
        case ActionType.GOTO:
            return f"{action.value.strip() if action.value else ''} -> GOTO"
        case ActionType.GO_BACK:
            return "GO BACK"
        case ActionType.GO_FORWARD:
            return "GO FORWARD"
        case ActionType.SELECT:
            return f"{target} -> SELECT \"{action.value.strip() if action.value else ''}\""
        case ActionType.NONE:
            return f"STOP -> {action.value}".strip()
        case ActionType.STOP:
            return f"STOP -> {action.value}".strip()
        case _:
            return f"{target} -> {action_type}"


@dataclass
class Mind2WebResultWriter:
    output_root: str
    task_id: Optional[str] = field(default=None, init=False)
    task: Optional[str] = field(default=None, init=False)
    start_url: Optional[str] = field(default=None, init=False)
    input_image_paths: Optional[List[str]] = field(default=None, init=False)
    trajectory_dir: Optional[str] = field(default=None, init=False)
    action_history: List[str] = field(default_factory=list, init=False)
    thoughts: List[str] = field(default_factory=list, init=False)
    action_raw_input_output: List[dict] = field(default_factory=list, init=False)
    screenshot_idx: int = field(default=0, init=False)

    def __post_init__(self):
        os.makedirs(self.output_root, exist_ok=True)

    def start_task(self, task_id: str, task: str, start_url: str | None = None, input_image_paths: list[str] | None = None):
        self.task_id = task_id
        self.task = task
        self.start_url = start_url
        self.input_image_paths = input_image_paths
        self.action_history = []
        self.thoughts = []
        self.action_raw_input_output = []
        self.screenshot_idx = 0
        self.trajectory_dir = os.path.join(self.output_root, task_id, "trajectory")
        os.makedirs(self.trajectory_dir, exist_ok=True)

    def _ensure_started(self):
        if self.task_id is None or self.trajectory_dir is None:
            raise RuntimeError("Mind2WebResultWriter.start_task must be called before logging steps.")

    def log_initial_state(self, state: StateInfo) -> str:
        self._ensure_started()
        return self._save_state_image(state.raw_state.screenshot)

    def log_step(self, low_level_task: LowLevelTask, state_after: StateInfo | None) -> str | None:
        self._ensure_started()
        self.action_history.append(format_action_for_mind2web(low_level_task))
        if low_level_task.reasoning:
            self.thoughts.append(low_level_task.reasoning)
        elif low_level_task.task:
            self.thoughts.append(str(low_level_task.task))

        step_idx = len(self.action_raw_input_output)
        raw_record = {}
        action = low_level_task.action
        if getattr(action, 'raw_input_messages', None) is not None:
            raw_record["input_msg"] = _save_messages_images_to_disk(
                action.raw_input_messages, self.trajectory_dir, step_idx
            )
        if getattr(action, 'raw_response', None) is not None:
            raw_record["raw_response"] = action.raw_response
        if getattr(action, 'grounding_info', None) is not None:
            raw_record["grounding_info"] = action.grounding_info
        self.action_raw_input_output.append(raw_record)

        if state_after is None:
            return None
        path = self._save_state_image(state_after.raw_state.screenshot)
        self._save_incremental_result()
        return path

    def _save_incremental_result(self):
        try:
            task_dir = os.path.join(self.output_root, self.task_id)
            os.makedirs(task_dir, exist_ok=True)
            result_path = os.path.join(task_dir, "result.json")
            result = {
                "task_id": self.task_id,
                "task": self.task,
                "start_url": self.start_url,
                "action_history": self.action_history,
                "thoughts": self.thoughts,
                "final_result_response": f"Task status: in_progress | Steps completed: {len(self.action_history)}",
                "input_image_paths": self.input_image_paths if self.input_image_paths is not None else [],
                "action_raw_input_output": self.action_raw_input_output,
            }
            with open(result_path, "w", encoding="utf-8") as f:
                f.write(_json_dumps(result))
        except Exception:
            pass

    def _save_state_image(self, screenshot: np.ndarray) -> str:
        filename = f"{self.screenshot_idx}_full_screenshot.png"
        self.screenshot_idx += 1
        path = os.path.join(self.trajectory_dir, filename)
        return _save_png(screenshot, path)

    def finalize(self, task_status: str, final_summary: str | None, final_result_response: str | None = None) -> str:
        self._ensure_started()
        if final_result_response is None:
            pieces = [f"Task status: {task_status}"]
            if final_summary:
                pieces.append(f"Summary: {final_summary}")
            final_result_response = " | ".join(pieces)

        task_dir = os.path.join(self.output_root, self.task_id)
        os.makedirs(task_dir, exist_ok=True)
        result_path = os.path.join(task_dir, "result.json")
        result = {
            "task_id": self.task_id,
            "task": self.task,
            "start_url": self.start_url,
            "action_history": self.action_history,
            "thoughts": self.thoughts,
            "final_result_response": final_result_response,
            "input_image_paths": self.input_image_paths if self.input_image_paths is not None else [],
            "action_raw_input_output": self.action_raw_input_output,
        }
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(_json_dumps(result))
        return result_path


def _save_messages_images_to_disk(messages: list, save_dir: str, step_idx: int) -> list:
    import base64

    img_counter = 0
    result = []
    for msg in messages:
        if not isinstance(msg, dict):
            result.append(msg)
            continue
        new_msg = {}
        for k, v in msg.items():
            if k == "content" and isinstance(v, list):
                new_content = []
                for item in v:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        detail = item.get("image_url", {}).get("detail", "auto")
                        if url.startswith("data:image/"):
                            try:
                                header, b64data = url.split(",", 1)
                                ext = "png"
                                if "jpeg" in header or "jpg" in header:
                                    ext = "jpg"
                                img_path = os.path.join(save_dir, f"step{step_idx}_input_img{img_counter}.{ext}")
                                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                                with open(img_path, "wb") as f:
                                    f.write(base64.b64decode(b64data))
                                new_content.append({"type": "image_url", "image_url": {"url": img_path, "detail": detail}})
                                img_counter += 1
                            except Exception:
                                new_content.append({"type": "image_url", "image_url": {"url": url[:100] + "...<save_failed>", "detail": detail}})
                        else:
                            new_content.append(item)
                    else:
                        new_content.append(item)
                new_msg[k] = new_content
            else:
                new_msg[k] = v
        result.append(new_msg)
    return result



