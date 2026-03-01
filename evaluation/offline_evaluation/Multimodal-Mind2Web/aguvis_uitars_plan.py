# -*- coding: utf-8 -*-
import argparse
import json
import io
import os
import re
import base64
import asyncio
import random
import math
from typing import Any, Dict, Tuple, Optional

from PIL import Image
from openai import AsyncOpenAI
import openai
from tqdm.asyncio import tqdm

# -----------------------------
# Qwen2.5-VL / UI-TARS coordinate resize utils (from UI-TARS coordinate guide)
# -----------------------------
IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor

def smart_resize(height: int, width: int,
                 factor: int = IMAGE_FACTOR,
                 min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> Tuple[int, int]:
    """Return (new_h, new_w) after Qwen2.5-VL style resizing."""
    if max(height, width) / max(1, min(height, width)) > MAX_RATIO:
        raise ValueError(f"absolute aspect ratio too large: {max(height, width) / min(height, width)}")

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def uitars_coord_to_original(model_x: int, model_y: int, orig_w: int, orig_h: int) -> Tuple[int, int]:
    """
    UI-TARS outputs coordinates on resized image resolution (new_w, new_h).
    Convert to original pixel coords using official mapping:
        x_orig = model_x / new_w * orig_w
        y_orig = model_y / new_h * orig_h
    """
    new_h, new_w = smart_resize(orig_h, orig_w)
    x = int(model_x / max(1, new_w) * orig_w)
    y = int(model_y / max(1, new_h) * orig_h)
    # clamp
    x = max(0, min(orig_w - 1, x))
    y = max(0, min(orig_h - 1, y))
    return x, y

def glm_coord_to_original(mx: int, my: int, orig_w: int, orig_h: int) -> Tuple[int, int]:
    """
    GLM grounding-style coordinates are relative values between 0 and 1000,
    normalized to the image size.
    """
    x = int(mx / 1000.0 * orig_w)
    y = int(my / 1000.0 * orig_h)
    x = max(0, min(orig_w - 1, x))
    y = max(0, min(orig_h - 1, y))
    return x, y


# -----------------------------
# Image encoding
# -----------------------------
def encode_image(path: str) -> Tuple[Optional[str], Tuple[int, int]]:
    try:
        with Image.open(path) as im:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode(), im.size  # (w, h)
    except Exception as e:
        print(f"Error encoding image {path}: {e}")
        return None, (0, 0)

# -----------------------------
# Prompts
# -----------------------------
# Your original JSON-action system prompt
SYSTEM_PROMPT_JSON = """You are a GUI agent. You are given a task and a screenshot of the screen. You need to choose actions from the the following list:
action_type: Click, value: None, point_2d: [x, y]
    ## Explanation: Tap or click a specific UI element and provide its coordinates

action_type: Select, value: Value to select, point_2d: [x, y] or None
    ## Explanation: Select an item from a list or dropdown menu

action_type: Write, value: Text to enter, point_2d: [x, y] or None
    ## Explanation: Enter text into a specific input field or at the current focus if coordinate is None

action_type: Scroll, value: "up" | "down" | "left" | "right", point_2d: None
    ## Explanation: Scroll a view or container in the specified direction
"""

# UI-TARS official prompt templates (compressed but faithful to prompt.py)
UITARS_COMPUTER_USE = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.
## Output Format
Thought: ...
Action: ...
## Action Space
click(point='x1 y1')
left_double(point='x1 y1')
right_single(point='x1 y1')
drag(start_point='x1 y1', end_point='x2 y2')
hotkey(key='ctrl c')
type(content='xxx')
scroll(point='x1 y1', direction='down or up or right or left')
wait()
finished(content='xxx')
## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.
## User Instruction
{instruction}
"""

UITARS_MOBILE_USE = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.
## Output Format
Thought: ...
Action: ...
## Action Space
click(point='x1 y1')
long_press(point='x1 y1')
type(content='')
scroll(point='x1 y1', direction='down or up or right or left')
open_app(app_name='')
drag(start_point='x1 y1', end_point='x2 y2')
press_home()
press_back()
finished(content='xxx')
## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.
## User Instruction
{instruction}
"""

UITARS_GROUNDING = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.
## Output Format
Action: ...
## Action Space
click(point='x1 y1')
## User Instruction
{instruction}
"""

QUESTION_DESCRIPTION = (
    "Please generate the next move according to the UI screenshot {img_size_string}, instruction and previous actions.\n\n"
    "Instruction: {task}\n\n"
    "Interaction History: {history}\n"
)

def build_prompt(agent_family: str,
                 uitars_template: str,
                 language: str,
                 task_desc: str,
                 prev_actions: list,
                 img_w: int,
                 img_h: int,
                 add_template: int,
                 reasoning: int,
                 model_name: str) -> Tuple[str, str]:
    prev_txt = "".join(f"\nStep {i+1}\n Action: {txt}\n" for i, txt in enumerate(prev_actions or []))
    img_size_string = f"(original image size {img_w}x{img_h})"

    base_user = QUESTION_DESCRIPTION.format(
        img_size_string=img_size_string,
        task=task_desc,
        history=prev_txt
    )

    # ---- UI-TARS ----
    if agent_family == "uitars":
        instruction = base_user
        if uitars_template == "computer":
            system_text = UITARS_COMPUTER_USE.format(language=language, instruction=instruction)
        elif uitars_template == "mobile":
            system_text = UITARS_MOBILE_USE.format(language=language, instruction=instruction)
        else:
            system_text = UITARS_GROUNDING.format(instruction=instruction)
        return system_text, ""

    # ---- AGUVIS ----
    if agent_family == "aguvis":
        system_text = (
            "You are an autonomous GUI agent. You are given an instruction, a screenshot, and action history. "
            "You MUST output an executable action in python using pyautogui.\n"
            "Output format (exactly):\n"
            "Thought: ...\n"
            "Action: ...\n"
            "pyautogui.<function>(...)\n"
        )
        return system_text, base_user

    # ---- GLM (NEW) ----
    if agent_family == "glm":
        # Recommend GLM-friendly format: keep <think> + <answer>{json}</answer>,
        # but ask point_2d in GLM grounding scale (0~1000).
        system_text = SYSTEM_PROMPT_JSON
        user_text = base_user

        # Force a stricter schema for GLM to stabilize parsing.
        user_text += (
            "\n\nSTRICT FORMAT:\n"
            "1) If you output <answer>, you MUST close it with </answer>.\n"
            "2) The assistant's final output MUST end with </answer> (no extra chars after).\n"
            "3) JSON must be valid.\n"
            "4) point_2d MUST be the center point [x, y] in GLM grounding scale (0~1000).\n"
        )
        return system_text, user_text

    # ---- default JSON family ----
    system_text = SYSTEM_PROMPT_JSON
    user_text = base_user

    # IMPORTANT: disable schema template for uitars/aguvis by design;
    # here only applies to JSON family.
    if add_template:
        if reasoning:
            if "Qwen3" in model_name:
                user_text += (
                    "\nThe response should be structured in the following format:\n"
                    "Thought: Your step-by-step thought process here...\n"
                    "<answer>\n{\n"
                    '  "action_description": "...",\n'
                    '  "action_type": "Click|Write|Scroll|Answer|...",\n'
                    '  "action_target": "...",\n'
                    '  "value": "None|...|up|down|left|right",\n'
                    '  "point_2d": [x, y]\n'
                    "}\n</answer>"
                )
            else:
                user_text += (
                    "\nThe response should be structured in the following format:\n"
                    "<think> ... </think>\n"
                    "<answer>\n{\n"
                    '  "action_description": "...",\n'
                    '  "action_type": "Click|Write|Scroll|Answer|...",\n'
                    '  "action_target": "...",\n'
                    '  "value": "None|...|up|down|left|right",\n'
                    '  "point_2d": [x, y]\n'
                    "}\n</answer>"
                )
        else:
            user_text += (
                "\nThe response should be structured in the following format:\n"
                "<answer>\n{\n"
                '  "action_description": "...",\n'
                '  "action_type": "Click|Write|Scroll|Answer|...",\n'
                '  "action_target": "...",\n'
                '  "value": "None|...|up|down|left|right",\n'
                '  "point_2d": [x, y]\n'
                "}\n</answer>"
            )

    return system_text, user_text

# -----------------------------
# Model call
# -----------------------------
async def get_llm_response(client, model, messages, temperature=0.5):
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=2048,
        )
        return response.choices[0].message.content
    except openai.APIError as e:
        print(f"Request failed: {e}")
        return None

# -----------------------------
# Parsers
# -----------------------------
def parse_json_action(response: str) -> Dict[str, Any]:
    out = {"action_type": "", "action_target": "", "value": "", "point_2d": [-100, -100], "reason": ""}

    think_match = re.search(r"<think>(.*?)</think>", response or "", re.DOTALL)
    if think_match:
        out["reason"] = think_match.group(1).strip()

    ans_match = re.search(r"<answer>\s*(\{.*?\})\s*</answer>", response or "", re.DOTALL)
    if not ans_match:
        brace = re.search(r"(\{.*\})", response or "", re.DOTALL)
        if not brace:
            return out
        ans_str = brace.group(1)
    else:
        ans_str = ans_match.group(1)

    try:
        obj = json.loads(ans_str)
        out["action_type"] = (obj.get("action_type") or "").strip()
        out["action_target"] = (obj.get("action_target") or "").strip()
        out["value"] = str(obj.get("value") if obj.get("value") is not None else "").strip()
        p = obj.get("point_2d", [-100, -100])
        if isinstance(p, (list, tuple)) and len(p) == 2:
            out["point_2d"] = [int(p[0]), int(p[1])]
    except Exception:
        return out

    return out


def _extract_first_json_object(text: str) -> str:
    """Extract the first {...} object using brace matching."""
    if not text:
        return ""
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return ""

def parse_json_action_robust(response: str) -> Dict[str, Any]:
    out = {
        "action_type": "",
        "action_target": "",
        "value": "",
        "point_2d": [-100, -100],
        "reason": "",
    }
    resp = response or ""

    # ---- reason: tolerate missing </think> ----
    if "<think>" in resp:
        after = resp.split("<think>", 1)[1]
        if "</think>" in after:
            out["reason"] = after.split("</think>", 1)[0].strip()
        elif "<answer>" in after:
            out["reason"] = after.split("<answer>", 1)[0].strip()
        else:
            out["reason"] = after.strip()

    # ---- locate answer zone (tolerate missing </answer>) ----
    zone = resp
    if "<answer>" in resp:
        zone = resp.split("<answer>", 1)[1]

    obj_str = _extract_first_json_object(zone)
    if not obj_str:
        obj_str = _extract_first_json_object(resp)
        if not obj_str:
            return out

    # ---- parse: json first; if fails, python literal fallback ----
    obj = None
    try:
        obj = json.loads(obj_str)  # strict JSON
    except Exception:
        try:
            obj = ast.literal_eval(obj_str)  # Python literal: supports None/True/False
        except Exception:
            # last resort: normalize common tokens then json.loads
            normalized = re.sub(r"\bNone\b", "null", obj_str)
            normalized = re.sub(r"\bTrue\b", "true", normalized)
            normalized = re.sub(r"\bFalse\b", "false", normalized)
            try:
                obj = json.loads(normalized)
            except Exception:
                return out

    if not isinstance(obj, dict):
        return out

    out["action_type"] = str(obj.get("action_type") or "").strip()
    out["action_target"] = str(obj.get("action_target") or "").strip()

    v = obj.get("value", "")
    # 兼容 value=None / "None" / null
    out["value"] = "" if v is None else str(v).strip()

    p = obj.get("point_2d", [-100, -100])
    if isinstance(p, (list, tuple)) and len(p) == 2:
        try:
            out["point_2d"] = [int(p[0]), int(p[1])]
        except Exception:
            pass

    return out

def _extract_uitars_action_line(resp: str) -> str:
    # Prefer "Action: ..." line
    m = re.search(r"^\s*Action:\s*(.+)\s*$", resp or "", flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    # fallback: something like "Action: click(...)" on same line
    m = re.search(r"Action:\s*(click|left_double|right_single|drag|hotkey|type|scroll|wait|finished)\(.*?\)", resp or "", re.DOTALL)
    return m.group(0).strip() if m else ""

def parse_uitars_action(response: str, orig_w: int, orig_h: int) -> Dict[str, Any]:
    """
    Parse UI-TARS action syntax from prompt.py (click(point='x y'), scroll(...), type(...), finished(...)).
    Convert coordinates back to original pixel coords using smart_resize mapping.
    """
    out = {"action_type": "", "action_target": "", "value": "", "point_2d": [-100, -100], "reason": ""}

    thought = re.search(r"^\s*Thought:\s*(.+)\s*$", response or "", flags=re.MULTILINE)
    if thought:
        out["reason"] = thought.group(1).strip()

    action_str = _extract_uitars_action_line(response or "")
    if not action_str:
        return out

    # Normalize
    # action_str might be "Action: click(...)" or "click(...)"
    action_str = re.sub(r"^\s*Action:\s*", "", action_str).strip()

    # Identify action name
    name_m = re.match(r"^([a-z_]+)\(", action_str)
    if not name_m:
        return out
    name = name_m.group(1)

    def parse_point(field: str='') -> Optional[Tuple[int, int]]:
        # point='x y' OR start_point='x y'
        m = re.search(rf"\s*'?\(?\s*(\d+)[,\s]+(\d+)\s*\)?'?", action_str)
        if not m:
            return None
        mx, my = int(m.group(1)), int(m.group(2))
        return uitars_coord_to_original(mx, my, orig_w, orig_h)

    if name.lower() in ("click", "left_double", "right_single", "long_press"):
        if name.lower() == "long_press":
            out["action_type"] = "LongPress"
        else:
            out["action_type"] = "Click"
        pt = parse_point()
        if pt:
            out["point_2d"] = [pt[0], pt[1]]
        out["value"] = "None"
    
    elif name.lower() == 'select':
        out["action_type"] = "Select"
        sm = re.search(r"value\s*=\s*'([^']+)'", action_str)
        out["value"] = sm.group(1) if sm else ""
        pt = parse_point()
        if pt:
            out["point_2d"] = [pt[0], pt[1]]

    elif name == "drag":
        out["action_type"] = "Drag"
        sp = parse_point("start_point")
        ep = parse_point("end_point")
        out["value"] = f"start={sp}, end={ep}"

    elif name == "hotkey":
        out["action_type"] = "KeyboardPress"
        km = re.search(r"key\s*=\s*'([^']+)'", action_str)
        out["value"] = km.group(1) if km else ""

    elif name == "type":
        out["action_type"] = "Write"
        cm = re.search(r"content\s*=\s*'(.+?)'\s*\)", action_str, re.DOTALL)
        out["value"] = cm.group(1) if cm else ""
        out["point_2d"] = [-100, -100]

    elif name == "scroll":
        out["action_type"] = "Scroll"
        dm = re.search(r"direction\s*=\s*'([^']+)'", action_str)
        out["value"] = (dm.group(1) if dm else "").strip()
        pt = parse_point("point")
        if pt:
            out["point_2d"] = [pt[0], pt[1]]

    elif name == "wait":
        out["action_type"] = "Wait"
        out["value"] = "5s"

    elif name == "finished":
        out["action_type"] = "Answer"
        cm = re.search(r"content\s*=\s*'(.+?)'\s*\)", action_str, re.DOTALL)
        out["value"] = cm.group(1) if cm else ""
        out["point_2d"] = [-100, -100]

    else:
        out["action_type"] = name
        out["value"] = action_str

    return out

def parse_aguvis_action(response: str, orig_w: int, orig_h: int) -> Dict[str, Any]:
    """
    Parse AGUVIS typical output:
      Thought: ...
      Action: ...
      pyautogui.click(x=0.52, y=0.42)
    """
    out = {"action_type": "", "action_target": "", "value": "", "point_2d": [-100, -100], "reason": ""}

    thought = re.search(r"^\s*Thought:\s*(.+)\s*$", response or "", flags=re.MULTILINE)
    if thought:
        out["reason"] = thought.group(1).strip()

    # pick last pyautogui.*(...) line
    lines = (response or "").splitlines()
    py_line = ""
    for ln in reversed(lines):
        if "pyautogui." in ln:
            py_line = ln.strip()
            break
    if not py_line:
        # fallback regex
        m = re.search(r"(pyautogui\.[a-zA-Z_]+\(.+?\))", response or "", re.DOTALL)
        py_line = m.group(1).strip() if m else ""

    if not py_line:
        return out

    fn_m = re.search(r"pyautogui\.([a-zA-Z_]+)\(", py_line)
    fn = fn_m.group(1) if fn_m else ""

    def parse_xy() -> Optional[Tuple[int, int]]:
        mx = re.search(r"\bx\s*=\s*([0-9]*\.?[0-9]+)", py_line)
        my = re.search(r"\by\s*=\s*([0-9]*\.?[0-9]+)", py_line)
        if not (mx and my):
            return None
        x = float(mx.group(1)); y = float(my.group(1))
        # normalized (0~1) -> pixel
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            px = int(x * orig_w)
            py = int(y * orig_h)
        else:
            px, py = int(x), int(y)
        px = max(0, min(orig_w - 1, px))
        py = max(0, min(orig_h - 1, py))
        return px, py

    if fn in ("click", "doubleClick", "rightClick", "leftClick"):
        out["action_type"] = "Click"
        pt = parse_xy()
        if pt:
            out["point_2d"] = [pt[0], pt[1]]
        out["value"] = "None"

    elif fn in ("write", "typewrite"):
        out["action_type"] = "Write"
        mm = re.search(r"message\s*=\s*'([^']*)'", py_line) or re.search(r"\(\s*'([^']*)'\s*\)", py_line)
        out["value"] = mm.group(1) if mm else ""

    elif fn == "press":
        out["action_type"] = "KeyboardPress"
        km = re.search(r"\(\s*'([^']+)'\s*\)", py_line)
        out["value"] = km.group(1) if km else ""

    elif fn == "hotkey":
        out["action_type"] = "KeyboardPress"
        keys = re.findall(r"'([^']+)'", py_line)
        out["value"] = " ".join(keys)

    elif fn == "scroll":
        out["action_type"] = "Scroll"
        # pyautogui.scroll(amount): negative -> down (common), positive -> up
        am = re.search(r"\(\s*([\-0-9]+)\s*\)", py_line)
        if am:
            amt = int(am.group(1))
            out["value"] = "down" if amt < 0 else "up"
        else:
            out["value"] = ""

    else:
        out["action_type"] = fn or "Unknown"
        out["value"] = py_line

    return out

# -----------------------------
# Main processing
# -----------------------------
async def process_row(row, sem, args, client, model_name, block_image_dir):
    async with sem:
        blocks_path   = row["blocks_path"]
        target_blocks = row.get("target_blocks", [])
        task_desc     = row["task"]
        if 'previous_actions_descriptions' in row:
            prev_actions  = row.get("previous_actions_descriptions", []) or []
        else:
            prev_actions  = row.get("previous_actions", []) or []

        block_num = 0
        response = ""
        parsed = {"action_type": "", "value": "", "point_2d": [-100, -100], "reason": ""}

        img_size = (0, 0)

        while True:
            img_file = os.path.join(block_image_dir, blocks_path, f"{block_num}.png")
            if not os.path.exists(img_file):
                break

            b64, img_size = encode_image(img_file)
            if not b64:
                break

            img_w, img_h = img_size[0], img_size[1]

            # Auto-disable add_template for uitars/aguvis to avoid format conflicts
            add_template = args.add_template
            if args.agent_family in ("uitars", "aguvis"):
                add_template = 0

            system_text, user_text = build_prompt(
                agent_family=args.agent_family,
                uitars_template=args.uitars_template,
                language=args.language,
                task_desc=task_desc,
                prev_actions=prev_actions,
                img_w=img_w,
                img_h=img_h,
                add_template=add_template,
                reasoning=args.reasoning,
                model_name=model_name
            )

            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_text}]},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
                    {"type": "text", "text": user_text}
                ]}
            ]

            response = await get_llm_response(client, model_name, messages, temperature=args.temperature)
            if response is None:
                break

            if args.agent_family == "uitars":
                parsed = parse_uitars_action(response, img_w, img_h)
            elif args.agent_family == "aguvis":
                parsed = parse_aguvis_action(response, img_w, img_h)
            else:
                # GLM postprocess: convert 0~1000 -> pixel
                if args.agent_family == "glm":
                    parsed = parse_json_action_robust(response + '</answer>')
                    px, py = parsed.get("point_2d", [-100, -100])
                    if isinstance(px, int) and isinstance(py, int) and 0 <= px <= 1000 and 0 <= py <= 1000:
                        x, y = glm_coord_to_original(px, py, img_w, img_h)
                        parsed["point_2d"] = [x, y]
                else:
                    parsed = parse_json_action_robust(response)

            # print(response, parsed)
            
            max_target = max(map(int, target_blocks)) if target_blocks else -1
            next_block_exists = os.path.exists(os.path.join(block_image_dir, blocks_path, f"{block_num+1}.png"))
            is_scroll_down = (parsed.get("action_type", "").lower() == "scroll" and str(parsed.get("value", "")).lower() == "down")

            if (not is_scroll_down) or (block_num > max_target) or (not next_block_exists):
                break
            block_num += 1

        row.update({
            "ans_block": block_num,
            "gpt_action": parsed.get("action_type", ""),
            "gpt_value": parsed.get("value", ""),
            "point_2d": parsed.get("point_2d", [-100, -100]),
            "response": response,
            "reason": parsed.get("reason", ""),
            "image_width": img_size[0],
            "image_height": img_size[1],
        })
        return row



async def main(args):
    sem = asyncio.Semaphore(args.sem_limit)

    with open(args.input_file) as f:
        rows = [json.loads(line) for line in f]

    tasks = [asyncio.create_task(process_row(r, sem, args, client, model_name, block_image_dir)) for r in rows]
    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        results.append(await coro)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as out:
        for r in results:
            out.write(json.dumps(r) + "\n")

    print(f"Done – saved {len(results)} rows to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GUI grounding/planning with OpenAI-compatible vLLM endpoint.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Model name passed to /v1/chat/completions")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--blocks", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--add_template", type=int, default=0)
    parser.add_argument("--reasoning", type=int, default=1)
    parser.add_argument("--sem_limit", type=int, default=64)

    # NEW: which “agent family” to adapt to
    parser.add_argument("--agent_family", type=str, default="json",
                    choices=["json", "uitars", "aguvis", "glm"])
    # NEW: UI-TARS prompt template selection
    parser.add_argument("--uitars_template", type=str, default="computer", choices=["computer", "mobile", "grounding"])
    # NEW: UI-TARS requires specifying thought language in prompt
    parser.add_argument("--language", type=str, default="English")

    args = parser.parse_args()

    block_image_dir = args.blocks
    model_name = args.model

    client = AsyncOpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key="token-abc123",
    )

    asyncio.run(main(args))
