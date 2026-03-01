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
import ast
from typing import Any, Dict, Tuple, Optional

from PIL import Image
import openai
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import aiofiles

def extract_glm_box(text: str) -> str | None:
    m = re.search(
        r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>",
        text,
        flags=re.DOTALL
    )
    return m.group(1).strip() if m else None

def parse_glm_kv_box(box: str):
    out = {
        "action_type": "",
        "action_target": "",
        "value": "",
        "point_2d": [-100, -100],
        "reason": "",
    }

    # action_type
    m = re.search(r"action_type\s*:\s*([A-Za-z_]+)", box)
    if m:
        out["action_type"] = m.group(1)

    # action_target
    m = re.search(r"action_target\s*:\s*['\"](.+?)['\"]", box)
    if m:
        out["action_target"] = m.group(1)

    # value
    m = re.search(r"value\s*:\s*([^,\n]+)", box)
    if m:
        out["value"] = m.group(1).strip()

    # point_2d (可能是 box)
    m = re.search(r"point_2d\s*:\s*\[([^\]]+)\]", box)
    if m:
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
        if len(nums) == 2:
            out["point_2d"] = nums
        elif len(nums) == 4:
            # box → center point
            x1, y1, x2, y2 = nums
            out["point_2d"] = [(x1 + x2) // 2, (y1 + y2) // 2]

    return out

def parse_glm_action(response: str):
    out = {
        "action_type": "",
        "action_target": "",
        "value": "",
        "point_2d": [-100, -100],
        "reason": "",
    }

    # reason from <think>
    m = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if m:
        out["reason"] = m.group(1).strip()

    box = extract_glm_box(response)
    if not box:
        return out

    parsed = parse_glm_kv_box(box)
    out.update(parsed)
    return out

# -----------------------------
# UI-TARS coordinate mapping (README: /1000 -> [0,1] -> pixel)
# -----------------------------
def uitars_1000_to_pixel(x1000: int, y1000: int, w: int, h: int) -> Tuple[int, int]:
    # README: x_rel = x/1000, x_abs = x_rel * width  :contentReference[oaicite:2]{index=2}
    x = int(round(w * (x1000 / 1000.0)))
    y = int(round(h * (y1000 / 1000.0)))
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
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
# Base system prompt (your JSON action space)
# -----------------------------
SYSTEM_PROMPT_JSON = (
    "You are a GUI agent. You are given a task and a screenshot of the screen. "
    "You need to perform a series of actions to complete the task. "
    "You need to choose actions from the the following list:\n"
    "action_type: Click, action_target: Element description, value: None, point_2d: [x, y]\n"
    "    ## Explanation: Tap or click a specific UI element and provide its coordinates\n\n"
    "action_type: Write, action_target: Element description or None, value: Text to enter, point_2d: [x, y]\n"
    "    ## Explanation: Enter text into a specific input field or at the current focus if coordinate is None\n\n"
    "action_type: LongPress, action_target: Element description, value: None, point_2d: [x, y]\n"
    "    ## Explanation: Press and hold on a specific UI element (mobile only) and provide its coordinates\n\n"
    "action_type: Scroll, action_target: None, value: \"up\" | \"down\" | \"left\" | \"right\", point_2d: None\n"
    "    ## Explanation: Scroll a view or container in the specified direction\n\n"
    "action_type: Wait, action_target: None, value: Number of seconds, point_2d: None\n"
    "    ## Explanation: Pause execution to allow the UI to load or update\n\n"
    "action_type: NavigateBack, action_target: None, value: None, point_2d: None\n"
    "    ## Explanation: Press the system \"Back\" button\n\n"
    "action_type: OpenApp, action_target: None, value: App name, point_2d: None\n"
    "    ## Explanation: Launch an app by its name (mobile only)\n\n"
    "action_type: Terminate, action_target: None, value: End-task message, point_2d: None\n"
    "    ## Explanation: Signal the end of the current task with a final message\n"
)

QUESTION_DESCRIPTION = (
    "Please generate the next move according to the UI screenshot {img_size_string}, instruction and previous actions.\n\n"
    "Instruction: {instruction}\n\n"
    "Interaction History: {history}\n"
)

# -----------------------------
# UI-TARS prompt templates (from README) :contentReference[oaicite:3]{index=3}
# -----------------------------
UITARS_PROMPT_MOBILE = r"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
Thought: ...
Action: ...

## Action Space
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
long_press(start_box='<|box_start|>(x1,y1)<|box_end|>', time='')
type(content='')
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
press_home()
press_back()
finished(content='') # Submit the task regardless of whether it succeeds or fails.

## Note
- Use English in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
"""

UITARS_PROMPT_COMPUTER = r"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
Thought: ...
Action: ...

## Action Space
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use "\n" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.

## Note
- Use Chinese in `Thought` part.
- Summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
"""

# -----------------------------
# OpenAI-compatible call
# -----------------------------
async def get_llm_response(client, model, messages, temperature=0.5, max_completion_tokens=2048):
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        return response.choices[0].message.content
    except openai.APIError as e:
        print(f"Request failed: {e}")
        return None

# -----------------------------
# Robust JSON extraction/parsing (fix GLM: missing </answer>, None, etc.)
# -----------------------------
def _strip_code_fences(s: str) -> str:
    s = re.sub(r"```(?:json)?\s*", "", s)
    s = s.replace("```", "")
    return s

def find_json_blocks(text: str):
    blocks = []
    stack = []
    start = None

    for i, ch in enumerate(text):
        if ch == '{':
            if not stack:
                start = i
            stack.append('{')
        elif ch == '}':
            if stack:
                stack.pop()
                if not stack and start is not None:
                    blocks.append(text[start:i+1])
                    start = None
    return blocks

def _find_json_candidate(text: str) -> Optional[str]:
    if not text:
        return None
    t = _strip_code_fences(text)

    # Prefer <answer> ... </answer> but allow missing closing tag
    m = re.search(r"<answer>\s*(\{.*)", t, flags=re.DOTALL)
    if m:
        cand = m.group(1)
        # If has </answer>, cut it
        cand = re.split(r"</answer>", cand, maxsplit=1, flags=re.IGNORECASE)[0]
        return cand.strip()

    # Else: take the last {...} block (often the action JSON is near the end)
    # blocks = re.findall(r"\{(?:[^{}]|(?R))*\}", t, flags=re.DOTALL)  # recursive regex supported? not in py
    # Python re doesn't support (?R); fallback to a simpler heuristic:
    # blocks = re.findall(r"\{.*\}", t, flags=re.DOTALL)
    blocks = find_json_blocks(t)
    if blocks:
        return blocks[-1].strip()

    return None

def _json_sanitize(s: str) -> str:
    # Replace python-style literals to JSON literals (only when they appear as tokens)
    s2 = re.sub(r"\bNone\b", "null", s)
    s2 = re.sub(r"\bTrue\b", "true", s2)
    s2 = re.sub(r"\bFalse\b", "false", s2)
    # Remove trailing commas before } or ]
    s2 = re.sub(r",\s*([}\]])", r"\1", s2)
    return s2

def parse_json_action_robust(response: str) -> Dict[str, Any]:
    """
    Returns:
      {action_type, action_target, value, point_2d, reason}
    Tolerates:
      - missing </answer>
      - value: None / True / False
      - code fences
      - single quotes (via ast.literal_eval fallback)
    """
    out = {
        "action_type": "",
        "action_target": "",
        "value": "",
        "point_2d": [-100, -100],
        "reason": "",
    }

    if not response:
        return out

    # reason
    think = re.search(r"<think>(.*?)</think>", response, flags=re.DOTALL | re.IGNORECASE)
    if think:
        out["reason"] = think.group(1).strip()
    else:
        # GLM sometimes outputs "Thought: ..."
        thought = re.search(r"^\s*(Thought|Reason)\s*:\s*(.+)$", response, flags=re.MULTILINE)
        if thought:
            out["reason"] = thought.group(2).strip()

    cand = _find_json_candidate(response)
    if not cand:
        return out

    cand = _json_sanitize(cand)

    # Try strict JSON
    try:
        obj = json.loads(cand)
    except Exception:
        # Fallback: python dict style (single quotes / None)
        try:
            obj = ast.literal_eval(cand)
        except Exception:
            return out

    if not isinstance(obj, dict):
        return out

    out["action_type"] = str(obj.get("action_type", "")).strip()
    out["action_target"] = str(obj.get("action_target", "")).strip()

    v = obj.get("value", "")
    if v is None:
        out["value"] = "None"
    else:
        out["value"] = str(v)

    p = obj.get("point_2d", [-100, -100])
    if isinstance(p, (list, tuple)) and len(p) == 2:
        try:
            out["point_2d"] = [int(p[0]), int(p[1])]
        except Exception:
            out["point_2d"] = [-100, -100]

    return out

# -----------------------------
# UI-TARS action parser (start_box / end_box / press_back etc.)
# -----------------------------
def parse_uitars_action(response: str, img_w: int, img_h: int) -> Dict[str, Any]:
    out = {
        "action_type": "",
        "action_target": "",
        "value": "",
        "point_2d": [-100, -100],
        "reason": "",
    }
    if not response:
        return out

    m = re.search(r"^\s*Thought:\s*(.+)$", response, flags=re.MULTILINE)
    if m:
        out["reason"] = m.group(1).strip()

    m = re.search(r"^\s*Action:\s*(.+)$", response, flags=re.MULTILINE)
    action_line = m.group(1).strip() if m else ""
    if not action_line:
        # fallback: find first function-like token
        m = re.search(r"\b(click|long_press|type|scroll|press_back|press_home|wait|finished|left_double|right_single|drag|hotkey)\s*\(.*\)", response)
        action_line = m.group(0).strip() if m else ""

    if not action_line:
        return out

    fn = re.match(r"^\s*([a-z_]+)\s*\(", action_line)
    fn = fn.group(1) if fn else ""

    def _extract_xy_1000(s: str) -> Optional[Tuple[int, int]]:
        # matches (x,y) within <|box_start|>(x1,y1)<|box_end|>
        mm = re.search(r"\(\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\)", s)
        if not mm:
            return None
        return int(mm.group(1)), int(mm.group(2))

    def _point_from_start_box() -> Optional[Tuple[int, int]]:
        mm = re.search(r"start_box\s*=\s*'([^']+)'", action_line)
        if not mm:
            # sometimes no keyword, directly contains box token
            mm = re.search(r"'<\|box_start\|>.*?<\|box_end\|>'", action_line)
        if not mm:
            return None
        xy = _extract_xy_1000(mm.group(1))
        if not xy:
            return None
        return uitars_1000_to_pixel(xy[0], xy[1], img_w, img_h)

    if fn in ("click", "left_double", "right_single"):
        out["action_type"] = "Click"
        pt = _point_from_start_box()
        if pt:
            out["point_2d"] = [pt[0], pt[1]]
        out["value"] = "None"

    elif fn == "long_press":
        out["action_type"] = "LongPress"
        pt = _point_from_start_box()
        if pt:
            out["point_2d"] = [pt[0], pt[1]]
        out["value"] = "None"

    elif fn == "type":
        out["action_type"] = "Write"
        cm = re.search(r"content\s*=\s*'(.+?)'\s*\)\s*$", action_line, flags=re.DOTALL)
        out["value"] = cm.group(1) if cm else ""
        out["point_2d"] = [-100, -100]

    elif fn == "scroll":
        out["action_type"] = "Scroll"
        dm = re.search(r"direction\s*=\s*'([^']+)'", action_line)
        out["value"] = (dm.group(1) if dm else "").strip()
        pt = _point_from_start_box()
        if pt:
            out["point_2d"] = [pt[0], pt[1]]

    elif fn == "press_back":
        out["action_type"] = "NavigateBack"
        out["value"] = "None"
        out["point_2d"] = [-100, -100]

    elif fn == "press_home":
        out["action_type"] = "OpenApp"  # 或者你也可以定义成 NavigateHome
        out["value"] = "HOME"
        out["point_2d"] = [-100, -100]

    elif fn == "wait":
        out["action_type"] = "Wait"
        out["value"] = "5"
        out["point_2d"] = [-100, -100]

    elif fn == "finished":
        out["action_type"] = "Terminate"
        cm = re.search(r"content\s*=\s*'(.+?)'\s*\)\s*$", action_line, flags=re.DOTALL)
        out["value"] = cm.group(1) if cm else ""
        out["point_2d"] = [-100, -100]

    else:
        out["action_type"] = fn or "Unknown"
        out["value"] = action_line

    return out

# -----------------------------
# AGUVIS parser (heuristic: parse pyautogui line)
# -----------------------------
def parse_aguvis_action(response: str, img_w: int, img_h: int) -> Dict[str, Any]:
    out = {
        "action_type": "",
        "action_target": "",
        "value": "",
        "point_2d": [-100, -100],
        "reason": "",
    }
    if not response:
        return out

    m = re.search(r"^\s*Thought:\s*(.+)$", response, flags=re.MULTILINE)
    if m:
        out["reason"] = m.group(1).strip()

    lines = (response or "").splitlines()
    py_line = ""
    for ln in reversed(lines):
        if "pyautogui." in ln:
            py_line = ln.strip()
            break
    if not py_line:
        m = re.search(r"(pyautogui\.[a-zA-Z_]+\(.+?\))", response, flags=re.DOTALL)
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
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            px = int(x * img_w)
            py = int(y * img_h)
        else:
            px, py = int(x), int(y)
        px = max(0, min(img_w - 1, px))
        py = max(0, min(img_h - 1, py))
        return px, py

    if fn.lower() in ("click", "doubleclick", "rightclick", "leftclick"):
        out["action_type"] = "Click"
        pt = parse_xy()
        if pt:
            out["point_2d"] = [pt[0], pt[1]]
        out["value"] = "None"

    elif fn.lower() in ("write", "typewrite"):
        out["action_type"] = "Write"
        mm = re.search(r"message\s*=\s*'([^']*)'", py_line) or re.search(r"\(\s*'([^']*)'\s*\)", py_line)
        out["value"] = mm.group(1) if mm else ""

    elif fn.lower() in ("press", "hotkey"):
        out["action_type"] = "KeyboardPress"
        keys = re.findall(r"'([^']+)'", py_line)
        out["value"] = " ".join(keys)

    elif fn.lower() == "scroll":
        out["action_type"] = "Scroll"
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
# Prompt/message builder
# -----------------------------
def build_messages(args, instruction: str, prev_actions: list, img_w: int, img_h: int) -> list:
    prev_txt = "".join(f"\nStep {i+1}\n Action: {txt}\n" for i, txt in enumerate(prev_actions or []))
    img_size_string = f"(original image size {img_w}x{img_h})"
    query = QUESTION_DESCRIPTION.format(img_size_string=img_size_string, instruction=instruction, history=prev_txt)

    # glm/json family: strongly encourage machine-readable JSON
    if args.agent_family in ("json", "glm"):
        if args.add_template:
            # GLM容易不闭合tag：建议直接让它输出“纯 JSON”，你解析最稳
            if args.agent_family == "glm":
                query += (
            "\n\nSTRICT FORMAT:\n"
            "1) If you output <answer>, you MUST close it with </answer>.\n"
            "2) The assistant's final output MUST end with </answer> (no extra chars after).\n"
            "3) JSON must be valid.\n"
            "4) point_2d MUST be the center point [x, y] in GLM grounding scale (0~1000).\n"
                    "\nThe response should be structured in the following format:\n"
                        "<think> ... </think>\n"
                        "<answer>\n{\n"
                        '  "action_type": "Click|Write|LongPress|Scroll|Wait|NavigateBack|OpenApp|Terminate",\n'
                        '  "action_target": "...",\n'
                        '  "value": null,\n'
                        '  "point_2d": [x, y]\n'
                        "}\n</answer>\n"
                )
            else:
                # 你原来的 <think>/<answer> 也保留，但解析不依赖 </answer>
                if args.reasoning:
                    query += (
                        "\nThe response should be structured in the following format:\n"
                        "<think> ... </think>\n"
                        "<answer>\n{\n"
                        '  "action_type": "Click|Write|LongPress|Scroll|Wait|NavigateBack|OpenApp|Terminate",\n'
                        '  "action_target": "...",\n'
                        '  "value": null,\n'
                        '  "point_2d": [x, y]\n'
                        "}\n</answer>\n"
                        "IMPORTANT: value must be null (not None).\n"
                    )
                else:
                    query += (
                        "\nReturn ONLY:\n"
                        "<answer>{...}</answer>\n"
                        "IMPORTANT: value must be null (not None).\n"
                    )

        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_JSON}]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{args._b64}", "detail": "high"}},
                {"type": "text", "text": query},
            ]},
        ]

    # UI-TARS: README示例是把 prompt+instruction 放在 text 里，然后再给 image :contentReference[oaicite:4]{index=4}
    if args.agent_family == "uitars":
        base = UITARS_PROMPT_MOBILE if args.uitars_template == "mobile" else UITARS_PROMPT_COMPUTER
        # 把 history 拼进去，UI-TARS prompt里“task and action history”是自然语言部分
        full_text = base + query
        return [
            {"role": "user", "content": [
                {"type": "text", "text": full_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{args._b64}"}},
            ]},
        ]

    # AGUVIS: 没有强制规范公开，我这里给一个“pyautogui行”的约束
    if args.agent_family == "aguvis":
        sys = (
            "You are an autonomous GUI agent. You are given an instruction, a screenshot, and action history. "
            "You MUST output an executable action in python using pyautogui.\n"
            "Output format:\n"
            "Thought: ...\n"
            "Action: ...\n"
            "pyautogui.<function>(...)\n"
            "Prefer pyautogui.click(x=..., y=...) (x,y can be pixel or normalized 0~1).\n"
        )
        return [
            {"role": "system", "content": [{"type": "text", "text": sys}]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{args._b64}", "detail": "high"}},
                {"type": "text", "text": query},
            ]},
        ]

    raise ValueError(f"Unknown agent_family={args.agent_family}")


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
# Row processing
# -----------------------------
async def process_row(row, sem, args, client, model_name):
    async with sem:
        annotation_id = row["episode_id"]
        high_level_instruction = row["goal"]
        low_level_instruction = row["step_instruction"]
        prev_actions = row.get("previous_actions", []) or []

        instruction = high_level_instruction if args.level == "high" else low_level_instruction

        img_file = os.path.join(args.screenshot_dir, row["screenshot"])
        b64, (img_w, img_h) = encode_image(img_file)
        if not b64:
            return {
                "episode_id": annotation_id,
                "step": row["step"],
                "instruction": instruction,
                "action_type": "",
                "element_description": "",
                "value": "",
                "point_2d": [-100, -100],
                "reason": "",
                "response": "",
                "error": f"failed_to_encode_image: {img_file}",
            }

        # pass b64 via args (avoid duplicating function signature)
        args._b64 = b64
        messages = build_messages(args, instruction, prev_actions, img_w, img_h)

        response = await get_llm_response(client, model_name, messages, temperature=args.temperature)
        # print(response)
        # breakpoint()
        if response is None:
            response = ""

        # Parse
        if args.agent_family == "glm":
            parsed = parse_json_action_robust(response + '</answer>')
            px, py = parsed.get("point_2d", [-100, -100])
            if isinstance(px, int) and isinstance(py, int) and 0 <= px <= 1000 and 0 <= py <= 1000:
                x, y = glm_coord_to_original(px, py, img_w, img_h)
                parsed["point_2d"] = [x, y]
        elif args.agent_family == "uitars":
            parsed = parse_uitars_action(response, img_w, img_h)
        elif args.agent_family == "aguvis":
            parsed = parse_aguvis_action(response, img_w, img_h)
        else:
            parsed = parse_json_action_robust(response)

        return {
            "episode_id": annotation_id,
            "step": row["step"],
            "instruction": instruction,
            "action_type": parsed.get("action_type", ""),
            "element_description": parsed.get("action_target", ""),
            "value": parsed.get("value", ""),
            "point_2d": parsed.get("point_2d", [-100, -100]),
            "reason": parsed.get("reason", ""),
            "response": response,
            "image_width": img_w,
            "image_height": img_h,
        }

async def main(args):
    sem = asyncio.Semaphore(args.sem_limit)

    with open(args.input_file, "r") as f:
        rows = json.load(f)

    tasks = [asyncio.create_task(process_row(r, sem, args, client, model_name)) for r in rows]
    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        results.append(await coro)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    async with aiofiles.open(args.output_file, "w") as out:
        for r in results:
            await out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Done – saved {len(results)} rows to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GUI agent runner (json/glm/uitars/aguvis) via OpenAI-compatible vLLM endpoint.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--screenshot_dir", type=str, required=True)
    parser.add_argument("--level", type=str, default="high", choices=["high", "low"])
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--add_template", type=int, default=0)
    parser.add_argument("--reasoning", type=int, default=0)
    parser.add_argument("--sem_limit", type=int, default=64)

    # model family switches
    parser.add_argument("--agent_family", type=str, default="json", choices=["json", "glm", "uitars", "aguvis"])
    parser.add_argument("--uitars_template", type=str, default="mobile", choices=["computer", "mobile"])

    args = parser.parse_args()

    model_name = args.model
    client = AsyncOpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key="token-abc123",
    )

    asyncio.run(main(args))
