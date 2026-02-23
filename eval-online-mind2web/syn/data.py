from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
import json
import numpy as np
import copy
import time
import re
import os
from syn.tools import tools_hash, tools_ndarray_image_save
import hashlib
from loguru import logger

_screenshot_save_path: str | None = None


def set_screenshot_save_path(path: str):
    global _screenshot_save_path
    _screenshot_save_path = path


class ActionType(Enum):
    CLICK = "click"
    TYPE = "type"
    HOVER = "hover"
    SCROLL = "scroll"
    PRESS = "press"
    NONE = "none"
    GOTO = "goto"
    GO_BACK = "go_back"
    GO_FORWARD = "go_forward"
    STOP = "stop"
    REFLECT = "reflect"
    SELECT = "select"


class LowTaskStatus(Enum):
    BEGIN = "BEGIN"
    IN_PROGRESS = "IN_PROGRESS"
    END = "END"
    NOTACHIEVEABLE = "NOT_ACHIEVABLE"



@dataclass
class Element:
    union_bound: tuple[int, int, int, int]
    id: str
    name: str
    role: str
    accessibility_tree_content: str
    action_type: ActionType | None = None
    value: str | None = None

    def __init__(self, accessibility_tree_content: str, union_bound: tuple[int, int, int, int], element_id: str):
        self.accessibility_tree_content = accessibility_tree_content
        self.union_bound = tuple(union_bound)
        self.id = element_id

        if len(accessibility_tree_content) == 0:
            self.name = "empty"
            self.role = "empty"
            self.action_type = ActionType.NONE
        else:
            self.name = re.sub(r'^\[\d+\]\s*', '', accessibility_tree_content)
            self.role = self.name.split()[0]
            self.action_type = self.determine_action_type(self.role)

    @staticmethod
    def determine_action_type(role: str) -> ActionType | None:
        role_lower = role.lower()
        if role_lower in {
            'button', 'link', 'menuitem', 'menuitemcheckbox', 'menuitemradio',
            'option', 'tab', 'checkbox', 'radio', 'switch', 'treeitem', 'rowheader'
        }:
            return ActionType.CLICK
        elif role_lower in {'tooltip', 'menubar', 'menu', 'tablist'}:
            return ActionType.HOVER
        elif role_lower in {'textbox', 'searchbox', 'combobox', 'spinbutton'}:
            return ActionType.TYPE
        elif role_lower in {'slider', 'timer'}:
            return ActionType.PRESS
        elif role_lower in {
            'document', 'main', 'article', 'feed', 'region', 'group',
            'list', 'listbox', 'tree', 'treegrid', 'grid', 'gridcell',
            'rowgroup', 'row', 'log', 'search', 'table', 'scrollbar', 'progressbar'
        }:
            return ActionType.SCROLL
        return ActionType.NONE

    def is_need_a_value_input(self) -> bool:
        return self.action_type in {ActionType.TYPE, ActionType.PRESS, ActionType.SCROLL, ActionType.GOTO}

    @staticmethod
    def create_empty_element() -> "Element":
        return Element(accessibility_tree_content="", union_bound=(0, 0, 0, 0), element_id="")

    def __hash__(self):
        hash_str = f"{self.name}_{self.union_bound or None}"
        return tools_hash(hash_str)


@dataclass
class RawState:
    url: str
    accessibility_tree: str
    observation_metadata: dict[str, Any]
    screenshot: np.ndarray
    timestamp: float

    def __init__(self, url, accessibility_tree, observation_metadata, screenshot, timestamp: float | None = None):
        if url.endswith('/'):
            url = url[:-1]
        self.url = copy.deepcopy(url)
        self.accessibility_tree = copy.deepcopy(accessibility_tree)
        self.observation_metadata = copy.deepcopy(observation_metadata)
        self.screenshot = copy.deepcopy(screenshot)
        self.timestamp = timestamp if isinstance(timestamp, float) else time.time()

    def __hash__(self):
        hash_str = f"{self.url}_{self.accessibility_tree}"
        h = hashlib.sha256()
        h.update(hash_str.encode('utf-8'))
        img_bytes = self.screenshot.tobytes()
        h.update(img_bytes)
        return int(h.hexdigest(), 16)

    def hash_by_screenshot(self) -> int:
        img_bytes = self.screenshot.tobytes()
        h = hashlib.sha256()
        h.update(img_bytes)
        return int(h.hexdigest(), 16)

    def to_dict(self) -> dict:
        path = str(self.hash_by_screenshot()) + '.jpg'
        path = f"{_screenshot_save_path}/{path}"
        if not os.path.exists(path):
            tools_ndarray_image_save(self.screenshot, path)
        return {
            "url": self.url,
            "accessibility_tree": self.accessibility_tree,
            "observation_metadata": self.observation_metadata,
            "screenshot": path,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(data: dict) -> "RawState":
        from syn.tools import tools_load_png_rgba
        screenshot_path = data["screenshot"]
        if isinstance(screenshot_path, str) and os.path.exists(screenshot_path):
            screenshot = tools_load_png_rgba(screenshot_path)
        else:
            raise ValueError(f"Screenshot path does not exist: {screenshot_path}")
        return RawState(
            url=data["url"],
            accessibility_tree=data["accessibility_tree"],
            observation_metadata=data["observation_metadata"],
            screenshot=screenshot,
            timestamp=data["timestamp"]
        )

    def __eq__(self, other):
        if not isinstance(other, RawState):
            return False
        return (
            self.url == other.url and
            self.accessibility_tree == other.accessibility_tree and
            self.observation_metadata == other.observation_metadata
        )


@dataclass
class StateInfo:
    raw_state: RawState
    elements: list[Element]
    summary: str | None = None

    def should_terminate(self) -> bool:
        return len(self.elements) == 0

    def __str__(self):
        return f"StateInfo(URL={self.raw_state.url}, Elements={len(self.elements)}, Summary={self.summary})"

    def __hash__(self):
        return hash(self.raw_state)


@dataclass
class Action:
    action_type: ActionType
    target_element: Element | None
    value: str | None = None
    coordinates: Optional[Tuple[int, int]] = None

    def __init__(self, element: Element | None, value: str | None, action_type: ActionType | None, coordinates: Optional[Tuple[int, int]] = None):
        if element is None:
            assert action_type is not None, "If element is None, action_type must be provided"
            if coordinates is None:
                assert not self._is_required_element(action_type), "If element is None and no coordinates, action_type cannot be CLICK, TYPE or HOVER"
            self.action_type = action_type
        elif action_type is not None:
            element.action_type = action_type

        self.action_type = action_type
        self.target_element = element
        self.coordinates = coordinates
        self.raw_response: str | None = None
        self.raw_input_messages: list | None = None

        default_empty_value = " "
        if value is None:
            value = default_empty_value
        elif isinstance(value, str) and len(value) == 0:
            value = default_empty_value
        self.value = self._value_ascil(value)

        if self._is_required_value(self.action_type) and not self.value.strip():
            raise ValueError(f"Action {self.action_type} requires a value, but got an empty value\n{self}")
        elif not self._is_required_value(self.action_type) and self.value.strip():
            logger.warning(f"Action {self.action_type} does not require a value, but got={self.value}, thus reset to empty string")
            self.value = default_empty_value

    def __hash__(self):
        hash_str = str(self)
        return tools_hash(hash_str)

    @staticmethod
    def _is_required_value(action_type: ActionType) -> bool:
        return action_type in {ActionType.TYPE, ActionType.PRESS, ActionType.SCROLL, ActionType.GOTO, ActionType.STOP, ActionType.NONE, ActionType.SELECT}

    @staticmethod
    def _is_required_element(action_type: ActionType) -> bool:
        return action_type in {ActionType.CLICK, ActionType.HOVER, ActionType.TYPE, ActionType.SELECT}

    def get_action_str(self) -> str:
        match self.action_type:
            case ActionType.CLICK:
                if self.target_element:
                    return f"click [{self.target_element.id}]"
                elif self.coordinates:
                    return f"click at ({int(self.coordinates[0])}, {int(self.coordinates[1])})"
                return "click [unknown]"
            case ActionType.TYPE:
                text = self.value if self.value else ""
                if self.target_element:
                    return f"type [{self.target_element.id}] [{text}]"
                elif self.coordinates:
                    return f"type [{text}] at ({int(self.coordinates[0])}, {int(self.coordinates[1])})"
                return f"type [{text}]"
            case ActionType.HOVER:
                if self.target_element:
                    return f"hover [{self.target_element.id}]"
                elif self.coordinates:
                    return f"hover at ({int(self.coordinates[0])}, {int(self.coordinates[1])})"
                return "hover [unknown]"
            case ActionType.SCROLL:
                direction = self.value if self.value in ["up", "down"] else "down"
                return f"scroll [{direction}]"
            case ActionType.PRESS:
                key_comb = self.value if self.value else ""
                return f"press [{key_comb}]"
            case ActionType.NONE | ActionType.STOP:
                return f"none"
            case ActionType.GOTO:
                return f"goto [{self.value}]"
            case ActionType.GO_BACK:
                return f"go_back"
            case ActionType.GO_FORWARD:
                return f"go_forward"
            case ActionType.SELECT:
                option = self.value if self.value else ""
                if self.target_element:
                    return f"select [{self.target_element.id}] [{option}]"
                elif self.coordinates:
                    return f"select [{option}] at ({int(self.coordinates[0])}, {int(self.coordinates[1])})"
                return f"select [{option}]"
            case _:
                raise RuntimeError(f"Unknown action type: {self.action_type}, target element={self.target_element}")

    def __str__(self):
        if self.action_type is None:
            logger.warning("Action type is None (not assigned yet)")
            action_str = "NOT-ASSIGNED"
            target_str = self.target_element.accessibility_tree_content
            value_str = "None"
        elif self.action_type == ActionType.NONE:
            action_str = "none (non-interactive action such as summary)"
            value_str = f"{self.value}"
            target_str = "None"
        elif self.target_element is None and self.coordinates is not None:
            action_str = self.action_type.value
            target_str = f"coordinates ({int(self.coordinates[0])}, {int(self.coordinates[1])})"
            value_str = f"{self.value}" if self.value and self.value.strip() else "None"
        elif self.target_element is None:
            assert isinstance(self.value, str) and self.value.strip(), f"If target_element is None and no coordinates, value must be non-empty for action_type={self.action_type}"
            action_str = self.action_type.value
            target_str = "None"
            value_str = f"{self.value}"
        elif self.action_type is ActionType.TYPE:
            action_str = self.action_type.value
            target_str = self.target_element.accessibility_tree_content
            value_str = f"{self.value}"
        elif self.action_type in {ActionType.CLICK, ActionType.HOVER}:
            action_str = self.action_type.value
            target_str = self.target_element.accessibility_tree_content
            value_str = "None"
        elif self.action_type in {ActionType.GO_BACK, ActionType.GO_FORWARD, ActionType.GOTO}:
            action_str = self.action_type.value
            target_str = "None"
            value_str = f"{self.value}"
        elif self.action_type == ActionType.STOP:
            action_str = "stop"
            target_str = "None"
            value_str = f"{self.value}"
        elif self.action_type == ActionType.SELECT:
            action_str = self.action_type.value
            target_str = self.target_element.accessibility_tree_content
            value_str = f"{self.value}"
        else:
            raise RuntimeError(f"Unknown action type: {self.action_type}, target element={self.target_element}, Value='{self.value}'")
        return f"{{Action: '{action_str}', Target: '{target_str}', Value: '{value_str}'}}"

    def _value_ascil(self, input_value: str) -> str:
        if len(input_value) > 0:
            input_value = input_value.replace("\u2018", "'").replace("\u2019", "'")
            input_value = input_value.replace("\u201c", '"').replace("\u201d", '"')
            input_value = input_value.replace("\u2013", "-").replace("\u2014", "-")
            input_value = input_value.replace("\u2026", "...")
            input_value = input_value.replace("\u00a0", " ").replace("\u2003", " ")
            input_value = input_value.replace("\u2022", "*")
        return input_value


@dataclass
class LowLevelTask:
    task: str | None
    curr_state: StateInfo
    action: Action
    state_after: StateInfo | None = None
    task_status: LowTaskStatus = LowTaskStatus.NOTACHIEVEABLE
    reasoning: str | None = None

    def __hash__(self):
        hash_str = f"{self.task}_{hash(self.curr_state)}_{hash(self.action)}_{self.task_status.name}"
        return tools_hash(hash_str)

    def is_executed(self) -> bool:
        return self.state_after is not None


@dataclass
class HighLevelTask:
    task: str
    trajectories: list[LowLevelTask]
    start_url: str | None = None

    def __hash__(self):
        hash_str = f"{self.task}_" + "_".join([str(hash(t)) for t in self.trajectories])
        return tools_hash(hash_str)


