import json
import re
import math
from typing import Dict, List, Tuple, Optional
from loguru import logger
import json_repair
from syn.tools import tools_robust_json_loads


# --- smart_resize for Qwen2.5 VL models ---

IMAGE_FACTOR = 28
SMART_RESIZE_MIN_PIXELS = 3136
SMART_RESIZE_MAX_PIXELS = 2109744
SMART_RESIZE_MAX_RATIO = 200


def _round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = SMART_RESIZE_MIN_PIXELS,
    max_pixels: int = SMART_RESIZE_MAX_PIXELS,
) -> Tuple[int, int]:
    if max(height, width) / min(height, width) > SMART_RESIZE_MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {SMART_RESIZE_MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)

    return h_bar, w_bar


# --- GUIPivot prompts ---

GUIPIVOT_SYSTEM_PROMPT = """You are a GUI agent. You are given a task and a screenshot of the screen. You need to choose actions from the the following list:
action_type: Click, action_target: Element description, value: None, point_2d: [x, y]
    ## Explanation: Tap or click a specific UI element and provide its coordinates

action_type: Select, action_target: Element description, value: Value to select, point_2d: [x, y] or None
    ## Explanation: Select an item from a list or dropdown menu

action_type: Write, action_target: Element description or None, value: Text to enter, point_2d: [x, y] or None
    ## Explanation: Enter text into a specific input field or at the current focus if coordinate is None

action_type: KeyboardPress, action_target: None, value: Key name (e.g., "enter"), point_2d: None
    ## Explanation: Press a specified key on the keyboard

action_type: Scroll, action_target: None, value: "up" | "down" | "left" | "right", point_2d: None
    ## Explanation: Scroll a view or container in the specified direction

action_type: Answer, action_target: None, value: Answer, point_2d: None
    ## Explanation: Return the final answer to the user's question
"""

GUIPIVOT_USER_PROMPT_TEMPLATE = '''Please generate the next move according to the UI screenshot {screensize}, instruction and previous actions.

Instruction: {instruction}

Interaction History: {actions}
'''

GUIPIVOT_OUTPUT_FORMAT = """

The response should be structured in the following format, make sure the output between <answer> and </answer> is a valid JSON object. Regarding the key "point_2d", please provide the coordinates on the screen where the action is to be performed; if not applicable, use [-100, -100]:
<thinking>Your step-by-step thought process here...</thinking>
<answer>
{
  "action_description": "the description of the action to perform, summarized in one sentence",
  "action_type": "the type of action to perform. Please follow the system prompt for available actions.",
  "value": "the input text or direction ('up', 'down', 'left', 'right') for the 'scroll' action, if applicable; otherwise, use 'None'",
  "point_2d": [x, y]
}
</answer>
"""


def format_previous_actions_guipivot(previous_actions: List[str], last_k: int = 15) -> str:
    if not previous_actions:
        return "None"
    formatted = ""
    for i, operation in enumerate(previous_actions[-last_k:]):
        formatted += f"Step {i + 1}\n Action: {operation}\n"
    return formatted if formatted else "None"


def build_guipivot_messages(
    task: str,
    base64_image: str,
    image_width: int,
    image_height: int,
    previous_actions: List[str],
    last_k: int = 15,
) -> List[Dict]:
    img_size_string = f'(original image size {image_width}x{image_height})'
    actions_str = format_previous_actions_guipivot(previous_actions, last_k=last_k)
    user_prompt = GUIPIVOT_USER_PROMPT_TEMPLATE.format(
        screensize=img_size_string,
        instruction=task,
        actions=actions_str,
    )
    user_prompt += GUIPIVOT_OUTPUT_FORMAT

    messages = [
        {"role": "system", "content": GUIPIVOT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "high"
                    }
                },
                {"type": "text", "text": user_prompt}
            ]
        }
    ]
    return messages


def parse_plan_guipivot(plan: str) -> Tuple[str, str, str]:
    think_pattern = r"<think>(.*?)</think>"
    think_match = re.search(think_pattern, plan, re.DOTALL)
    thought = think_match.group(1).strip() if think_match else ""

    if not think_match:
        think_pattern = r"<thinking>(.*?)</thinking>"
        think_match = re.search(think_pattern, plan, re.DOTALL)
        thought = think_match.group(1).strip() if think_match else ""

    answer_match = re.search(r"<answer>(.*?)</answer>", plan, re.DOTALL)
    action = answer_match.group(1).strip() if answer_match else ""
    if not action:
        try:
            action = tools_robust_json_loads(plan)
            if isinstance(action, dict) and len(action) > 0:
                action = json.dumps(action)
            else:
                action = ""
        except Exception as e:
            action = ""
            logger.error(f"Failed to parse action as JSON: {e}, plan={plan}")

    operation = ""
    if action:
        op_match = re.search(r'"action_description"\s*:\s*"([^"]+)"', action)
        if op_match:
            operation = op_match.group(1)
        else:
            type_match = re.search(r'"action_type"\s*:\s*"([^"]+)"', action)
            if type_match:
                operation = type_match.group(1)

    return thought, operation, action


def _extract_fields_via_regex(raw: str) -> Dict:
    result = {}
    for field in ("action_type", "action_description", "value"):
        m = re.search(rf"<{field}>(.*?)</{field}>", raw, re.DOTALL)
        if m:
            val = m.group(1).strip()
            if val.lower() == "none":
                val = None
            result[field] = val

    p2d_match = re.search(r"<point_2d>\s*\[?\s*(-?\d+(?:\.\d+)?)\s*[,\s]+\s*(-?\d+(?:\.\d+)?)\s*\]?\s*</point_2d>", raw)
    if not p2d_match:
        p2d_match = re.search(r'"point_2d"\s*:\s*\[?\s*(-?\d+(?:\.\d+)?)\s*[,\s]+\s*(-?\d+(?:\.\d+)?)', raw)
    if p2d_match:
        result["point_2d"] = [float(p2d_match.group(1)), float(p2d_match.group(2))]

    if "action_type" not in result:
        at_match = re.search(r'"action_type"\s*:\s*"([^"]+)"', raw)
        if at_match:
            result["action_type"] = at_match.group(1)

    if "action_description" not in result:
        ad_match = re.search(r'"action_description"\s*:\s*"([^"]*)"', raw)
        if ad_match:
            result["action_description"] = ad_match.group(1)

    if "value" not in result:
        v_match = re.search(r'"value"\s*:\s*"([^"]*)"', raw)
        if v_match:
            result["value"] = v_match.group(1)

    return result


def parse_guipivot_json(json_str: str) -> Dict:
    try:
        data = json_repair.loads(json_str)
        if isinstance(data, dict) and len(data) > 0:
            return data
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and len(item) > 0:
                    return item
    except Exception:
        pass

    cleaned = json_str
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()

    if cleaned != json_str:
        try:
            data = json_repair.loads(cleaned)
            if isinstance(data, dict) and len(data) > 0:
                return data
        except Exception:
            pass

    fixed = json_str
    fixed = re.sub(r':\s*None\b', ': null', fixed)
    fixed = re.sub(r':\s*True\b', ': true', fixed)
    fixed = re.sub(r':\s*False\b', ': false', fixed)
    try:
        data = json.loads(fixed)
        if isinstance(data, dict) and len(data) > 0:
            return data
    except Exception:
        pass

    data = _extract_fields_via_regex(json_str)
    if data and "action_type" in data:
        return data

    raise ValueError(f"Failed to parse GUIPivot JSON with all strategies: {json_str[:300]}")


def _normalize_guipivot_action_type(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return raw

    s = raw.strip()
    for prefix in ("action_type:", "action:", "action_type :", "action :"):
        if s.lower().startswith(prefix):
            s = s[len(prefix):].strip()
            break

    key = s.lower().replace(" ", "").replace("_", "")

    CANONICAL_MAP = {
        "click": "Click",
        "leftclick": "Click",
        "left": "Click",
        "tap": "Click",
        "select": "Select",
        "write": "Write",
        "type": "Write",
        "input": "Write",
        "keyboardpress": "KeyboardPress",
        "keypress": "KeyboardPress",
        "press": "KeyboardPress",
        "scroll": "Scroll",
        "swipe": "Scroll",
        "scrolldown": "Scroll",
        "scrollup": "Scroll",
        "answer": "Answer",
        "back": "Back",
        "goback": "Back",
        "navigate": "Navigate",
        "openwebsite": "Navigate",
        "terminate": "terminate",
        "stop": "terminate",
        "wait": "wait",
        "call": "Answer",
    }

    return CANONICAL_MAP.get(key, s)


def convert_guipivot_to_exec_actions(
    data: Dict,
    screen_width: int,
    screen_height: int,
    model_coord_size: int = 1000,
    use_smart_resize: bool = False,
) -> List[Dict]:
    action_type = _normalize_guipivot_action_type(data.get("action_type", ""))
    value = data.get("value", "")
    point_2d = data.get("point_2d")

    if not point_2d or not isinstance(point_2d, list):
        action_target = data.get("action_target", {})
        if isinstance(action_target, dict):
            if "point_2d" in action_target:
                point_2d = action_target["point_2d"]
            elif "coordinates" in action_target:
                point_2d = action_target["coordinates"]
        if not point_2d or not isinstance(point_2d, list):
            for alt_key in ("coordinates", "position", "coord", "point"):
                alt = data.get(alt_key)
                if isinstance(alt, list) and len(alt) == 2:
                    point_2d = alt
                    break

    if value == "None" or value is None:
        value = ""

    def scale_coords(x, y):
        if x is None or y is None:
            return None, None
        if x == -100 and y == -100:
            return None, None
        if use_smart_resize:
            resize_h, resize_w = smart_resize(screen_height, screen_width)
            scaled_x = (x / resize_w) * screen_width
            scaled_y = (y / resize_h) * screen_height
        else:
            scaled_x = (x / model_coord_size) * screen_width
            scaled_y = (y / model_coord_size) * screen_height
        return scaled_x, scaled_y

    exec_actions = []

    try:
        if action_type in ["Click", "Select"]:
            if point_2d and isinstance(point_2d, list) and len(point_2d) == 2:
                x, y = scale_coords(point_2d[0], point_2d[1])
                if x is not None and y is not None:
                    exec_actions = [{"name": "click", "parameters": {"x": x, "y": y, "clicks": 1, "button": "left"}}]
                else:
                    raise ValueError(f"Invalid point_2d after scaling: {point_2d}")
            else:
                raise ValueError(f"Invalid point_2d: {point_2d}")

        elif action_type == "Write":
            if point_2d and isinstance(point_2d, list) and len(point_2d) == 2:
                x, y = scale_coords(point_2d[0], point_2d[1])
                if x is not None and y is not None:
                    exec_actions.append({"name": "click", "parameters": {"x": x, "y": y, "clicks": 1, "button": "left"}})
            exec_actions.append({"name": "write", "parameters": {"message": value}})
            exec_actions.append({"name": "press", "parameters": {"keys": "enter"}})

        elif action_type == "KeyboardPress":
            safe_value = value.lower() if value else ""
            exec_actions = [{"name": "press", "parameters": {"keys": safe_value}}]

        elif action_type == "Scroll":
            direction = value.lower() if value else "down"
            mapping = {"up": "down", "down": "up", "left": "right", "right": "left"}
            target_direction = mapping.get(direction, "down")
            exec_actions = [{"name": "swipe", "parameters": {"direction": target_direction, "amount": 0.5}}]

        elif action_type == "Back" or (action_type == "Navigate" and value == "Back"):
            exec_actions = [{"name": "back", "parameters": {}}]

        elif action_type == "Answer":
            safe_value = value if value else "Task Completed"
            exec_actions = [{"name": "response", "parameters": {"answer": safe_value}}]

        elif action_type == "terminate":
            safe_value = value if value else "Task Completed"
            exec_actions = [{"name": "terminate", "parameters": {"status": "success", "info": safe_value}}]

        else:
            logger.warning(f"Unknown action_type in guipivot JSON: {action_type}, data={data}")
            exec_actions = [{"name": "wait", "parameters": {"seconds": 1}}]

    except Exception as e:
        logger.error(f"Error converting guipivot action: {e}, data={data}")
        exec_actions = [{"name": "wait", "parameters": {"seconds": 1}}]

    return exec_actions


def guipivot_action_to_mind2web_str(exec_action: Dict) -> str:
    name = exec_action.get("name", "")
    params = exec_action.get("parameters", {})

    if name == "click":
        x, y = params.get("x"), params.get("y")
        if x is not None and y is not None:
            return f"CLICK at ({int(x)}, {int(y)})"
        return "CLICK"
    elif name == "write":
        message = params.get("message", "")
        return f"TYPE \"{message}\""
    elif name == "press":
        keys = params.get("keys", "")
        return f"PRESS {keys}"
    elif name == "swipe":
        direction = params.get("direction", "")
        display_mapping = {"up": "down", "down": "up", "left": "right", "right": "left"}
        display_dir = display_mapping.get(direction, direction)
        return f"SCROLL {display_dir}"
    elif name == "back":
        return "GO BACK"
    elif name == "response":
        answer = params.get("answer", "")
        return f"ANSWER -> {answer}"
    elif name == "terminate":
        info = params.get("info", "")
        return f"TERMINATE -> {info}"
    elif name == "wait":
        return "WAIT"
    else:
        return f"{name.upper()}"


# --- GUIPivot Grounding Mode prompts ---

GUIPIVOT_GROUNDING_SYSTEM_PROMPT = """You are a GUI agent. You are given a task and a screenshot of the screen. You need to choose actions from the the following list:
action_type: Click, action_target: Element description, value: None
    ## Explanation: Tap or click a specific UI element. Describe the target element clearly.

action_type: Select, action_target: Element description, value: Value to select
    ## Explanation: Select an item from a list or dropdown menu

action_type: Write, action_target: Element description or None, value: Text to enter
    ## Explanation: Enter text into a specific input field or at the current focus if target is None

action_type: KeyboardPress, action_target: None, value: Key name (e.g., "enter")
    ## Explanation: Press a specified key on the keyboard

action_type: Scroll, action_target: None, value: "up" | "down" | "left" | "right"
    ## Explanation: Scroll a view or container in the specified direction

action_type: Answer, action_target: None, value: Answer
    ## Explanation: Return the final answer to the user's question
"""

GUIPIVOT_GROUNDING_OUTPUT_FORMAT = """

The response should be structured in the following format, make sure the output between <answer> and </answer> is a valid JSON object. For the key "action_target", describe the UI element you want to interact with as precisely as possible so a visual grounding model can locate it:
<thinking>Your step-by-step thought process here...</thinking>
<answer>
{
  "action_description": "the description of the action to perform, summarized in one sentence",
  "action_type": "the type of action to perform. Please follow the system prompt for available actions.",
  "value": "the input text or direction ('up', 'down', 'left', 'right') for the 'scroll' action, if applicable; otherwise, use 'None'",
  "action_target": "a precise description of the UI element to interact with, e.g. 'the search input field at the top of the page', 'the blue Submit button'. Use 'None' if not applicable."
}
</answer>
"""


def build_guipivot_grounding_messages(
    task: str,
    base64_image: str,
    image_width: int,
    image_height: int,
    previous_actions: List[str],
    last_k: int = 15,
) -> List[Dict]:
    img_size_string = f'(original image size {image_width}x{image_height})'
    actions_str = format_previous_actions_guipivot(previous_actions, last_k=last_k)
    user_prompt = GUIPIVOT_USER_PROMPT_TEMPLATE.format(
        screensize=img_size_string,
        instruction=task,
        actions=actions_str,
    )
    user_prompt += GUIPIVOT_GROUNDING_OUTPUT_FORMAT

    messages = [
        {"role": "system", "content": GUIPIVOT_GROUNDING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "high"
                    }
                },
                {"type": "text", "text": user_prompt}
            ]
        }
    ]
    return messages


def build_uground_messages(description: str, base64_image: str) -> List[Dict]:
    """Build messages for UGround visual grounding model (OpenAI-compatible format)."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {
                    "type": "text",
                    "text": f"""
  Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.

  - Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
  - If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
  - Your answer should be a single string (x, y) corresponding to the point of the interest.

  Description: {description}

  Answer:"""
                },
            ],
        },
    ]


def parse_grounding_response(response_text: str) -> Tuple[float, float]:
    """Parse UGround response to extract (x, y) coordinates in [0, 1000) range."""
    text = response_text.strip()
    # Try to find (x, y) pattern
    match = re.search(r'\(?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)?', text)
    if match:
        return float(match.group(1)), float(match.group(2))
    # Try to find [x, y] pattern
    match = re.search(r'\[?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]?', text)
    if match:
        return float(match.group(1)), float(match.group(2))
    raise ValueError(f"Failed to parse grounding coordinates from: {text}")
