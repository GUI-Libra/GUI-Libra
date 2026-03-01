# -*- coding: utf-8 -*-
import argparse
import json
import io
import os
import re
import base64
import asyncio
import ast
from typing import Any, Dict, Tuple, Optional, List
import random
from PIL import Image
from openai import AsyncOpenAI
import openai
from tqdm.asyncio import tqdm


# =============================
# Image encoding
# =============================
def encode_image(path: str) -> Tuple[Optional[str], Tuple[int, int]]:
    try:
        with Image.open(path) as im:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode(), im.size  # (w, h)
    except Exception as e:
        print(f"Error encoding image {path}: {e}")
        return None, (0, 0)


# =============================
# Official Prompts
# =============================

def build_guir1_prompt(task_text: str, history_text: str) -> str:
    """
    GUI-R1 official prompt (Android)
    """
    text = (
        f"You are GUI-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, "
        f"I want you to continue executing the command '{task_text}', with the action history being '{history_text}'.\n"
        "Please provide the action to perform (enumerate from ['wait', 'click', 'type', 'scroll']), "
        "the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.\n"
        "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
        "<think> ... </think> <answer>[{'action': enum['wait', 'click', 'type', 'select', 'scroll'], "
        "'point': [x, y], 'input_text': 'no input text [default]'}]</answer>\n"
        "Note:\n specific input text (no default) is necessary for actions enum['type', 'scroll'] \n Example:\n"
        "[{'action': enum['wait', 'press_enter'], 'point': [-100, -100], 'input_text': 'no input text'}]\n"
        "[{'action': enum['click'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
        "[{'action': enum['type'], 'point': [123, 300], 'input_text': 'shanghai shopping mall'}]\n"
        "[{'action': enum['select'], 'point': [-100, -100], 'input_text': 'value to select'}]\n"
        "[{'action': enum['scroll'], 'point': [-100, -100], 'input_text': enum['up', 'left', 'right', 'down']}]"
    )
    # Their code does: text = '<image>\n' + text
    return "<image>\n" + text


def build_uir1_prompt(task_text: str) -> str:
    """
    UI-R1 official prompt (AndroidControl evaluation)
    """
    question_template = (
        f"In this UI screenshot, I want to perform the command '{task_text}'.\n"
        "Please provide the action to perform (enumerate in ['click', 'open_app', 'scroll', 'navigate_back', 'input_text]')"
        "and the coordinate where the cursor is moved to(integer) if click is performed.\n"
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
        "The output answer format should be as follows:\n"
        "<think> ... </think> <answer>[{'action': enum['click', 'open_app', 'scroll', 'navigate_back', 'input_text], "
        "'coordinate': [x, y]}]</answer>\n"
        "Please strictly follow the format."
    )
    return question_template


def build_prompt_r1_only(agent_family: str,
                         task_desc: str,
                         prev_actions: list) -> Tuple[str, str]:
    """
    Returns (system_text, user_text)
    Both GUI-R1 and UI-R1 use user-side prompt with <think>/<answer>.
    """
    # compact history for GUI-R1
    if prev_actions:
        history_text = " | ".join([str(x).strip() for x in prev_actions[-8:]])
    else:
        history_text = "None"

    if agent_family == "gui_r1":
        system_text = ""
        user_text = build_guir1_prompt(task_text=task_desc, history_text=history_text)
        return system_text, user_text

    if agent_family == "ui_r1":
        system_text = ""
        user_text = build_uir1_prompt(task_text=task_desc)
        return system_text, user_text

    raise ValueError(f"Unknown agent_family={agent_family}")


# =============================
# OpenAI-compatible vLLM call
# =============================
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


# =============================
# Parsers (list-of-dict in <answer>)
# =============================

def _extract_think(resp: str) -> str:
    m = re.search(r"<think>(.*?)</think>", resp or "", flags=re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_answer_payload(resp: str) -> str:
    """
    Extract inside <answer>...</answer>.
    Tolerate missing closing tag.
    """
    if not resp:
        return ""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", resp, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    if "<answer>" in resp:
        return resp.split("<answer>", 1)[1].strip()

    return resp.strip()


def _extract_first_list_literal(text: str) -> str:
    """
    Find first Python-like list literal: [ {...} ]
    """
    if not text:
        return ""
    m = re.search(r"(\[\s*\{.*?\}\s*\])", text, flags=re.DOTALL)
    if m:
        return m.group(1)
    # tolerate single dict
    m2 = re.search(r"(\{\s*.*?\s*\})", text, flags=re.DOTALL)
    if m2:
        return f"[{m2.group(1)}]"
    return ""


def parse_list_action(resp: str) -> List[Dict[str, Any]]:
    """
    Parse list-of-dict output from <answer>.
    Use ast.literal_eval for robustness.
    """
    payload = _extract_answer_payload(resp)
    list_str = _extract_first_list_literal(payload)
    if not list_str:
        return []

    try:
        obj = ast.literal_eval(list_str)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    except Exception:
        pass

    # last resort: try JSON load after basic normalization
    try:
        normalized = list_str.replace("None", "null").replace("True", "true").replace("False", "false")
        obj = json.loads(normalized)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    except Exception:
        return []

    return []


def parse_guir1_action(response: str) -> Dict[str, Any]:
    """
    GUI-R1 expected dict keys:
      action, point, input_text
    """
    out = {
        "action_type": "",
        "value": "",
        "point_2d": [-100, -100],
        "reason": _extract_think(response),
        "raw_action": "",
        "raw_input_text": "",
    }

    lst = parse_list_action(response)
    if not lst:
        return out

    d = lst[0]
    act = str(d.get("action", "")).strip().lower()
    pt = d.get("point", [-100, -100])
    it = d.get("input_text", "no input text")

    out["raw_action"] = act
    out["raw_input_text"] = "" if it is None else str(it).strip()

    # Map to unified schema
    if act == "click":
        out["action_type"] = "Click"
        if isinstance(pt, (list, tuple)) and len(pt) == 2:
            out["point_2d"] = [int(pt[0]), int(pt[1])]
        out["value"] = "None"

    elif act == "long_press":
        out["action_type"] = "LongPress"
        if isinstance(pt, (list, tuple)) and len(pt) == 2:
            out["point_2d"] = [int(pt[0]), int(pt[1])]
        out["value"] = "None"

    elif act == "type":
        out["action_type"] = "Write"
        out["value"] = out["raw_input_text"]
        if isinstance(pt, (list, tuple)) and len(pt) == 2:
            out["point_2d"] = [int(pt[0]), int(pt[1])]

    elif act == "open_app":
        out["action_type"] = "OpenApp"
        out["value"] = out["raw_input_text"]
        out["point_2d"] = [-100, -100]

    elif act == "scroll":
        out["action_type"] = "Scroll"
        # GUI-R1 scroll direction in input_text
        out["value"] = out["raw_input_text"].lower()
        out["point_2d"] = [-100, -100]
    elif act == "press_enter":
        out["action_type"] = "KeyboardPress"
        out["value"] = "enter"
        out["point_2d"] = [-100, -100]
    elif act == 'select':
        out["action_type"] = "Select"
        out["value"] = out["raw_input_text"]
        out["point_2d"] = [-100, -100]

    elif act == "press_back":
        out["action_type"] = "PressBack"
        out["value"] = "back"
        out["point_2d"] = [-100, -100]

    elif act == "wait":
        out["action_type"] = "Wait"
        out["value"] = "5s"
        out["point_2d"] = [-100, -100]

    else:
        out["action_type"] = act
        out["value"] = out["raw_input_text"]

    return out


def parse_uir1_action(response: str) -> Dict[str, Any]:
    """
    UI-R1 expected dict keys (from your template):
      action, coordinate

    Note: template doesn't specify direction/text explicitly.
    We robustly accept extra keys if the model outputs them (e.g., "text", "input_text", "direction").
    """
    out = {
        "action_type": "",
        "value": "",
        "point_2d": [-100, -100],
        "reason": _extract_think(response),
        "raw_action": "",
    }

    lst = parse_list_action(response)
    if not lst:
        return out

    d = lst[0]
    act = str(d.get("action", "")).strip().lower()
    coord = d.get("coordinate", [-100, -100])

    # optional extras (not in strict template, but sometimes appear)
    extra_text = d.get("text", None)
    extra_input = d.get("input_text", None)
    extra_dir = d.get("direction", None)

    out["raw_action"] = act

    if act == "click":
        out["action_type"] = "Click"
        if isinstance(coord, (list, tuple)) and len(coord) == 2:
            out["point_2d"] = [int(coord[0]), int(coord[1])]
        out["value"] = "None"

    elif act == "open_app":
        out["action_type"] = "OpenApp"
        # If model provides app name in extras, keep it; else empty
        if extra_text is not None:
            out["value"] = str(extra_text)
        elif extra_input is not None:
            out["value"] = str(extra_input)
        else:
            out["value"] = ""
        out["point_2d"] = [-100, -100]

    elif act == "scroll":
        out["action_type"] = "Scroll"
        # UI-R1 template doesn't include direction; for block navigation we assume "down"
        if extra_dir is not None:
            out["value"] = str(extra_dir).lower()
        elif extra_text is not None:
            out["value"] = str(extra_text).lower()
        elif extra_input is not None:
            out["value"] = str(extra_input).lower()
        else:
            out["value"] = "down"
        out["point_2d"] = [-100, -100]

    elif act == "navigate_back":
        out["action_type"] = "PressBack"
        out["value"] = "back"
        out["point_2d"] = [-100, -100]
    elif act == 'select':
        out["action_type"] = "Select"
        if extra_text is not None:
            out["value"] = str(extra_text)
        elif extra_input is not None:
            out["value"] = str(extra_input)
        else:
            out["value"] = ""
        out["point_2d"] = [-100, -100]
    
    elif act == 'keyboard_press':
        out["action_type"] = "KeyboardPress"
        if extra_text is not None:
            out["value"] = str(extra_text).lower()
        elif extra_input is not None:
            out["value"] = str(extra_input).lower()
        else:
            out["value"] = "enter"
        out["point_2d"] = [-100, -100]
    
    elif act == 'press_back':
        out["action_type"] = "PressBack"
        out["value"] = "back"
        out["point_2d"] = [-100, -100]


    elif act == "input_text":
        out["action_type"] = "Write"
        if extra_text is not None:
            out["value"] = str(extra_text)
        elif extra_input is not None:
            out["value"] = str(extra_input)
        else:
            out["value"] = ""  # template didn't define where text lives
        if isinstance(coord, (list, tuple)) and len(coord) == 2:
            out["point_2d"] = [int(coord[0]), int(coord[1])]

    else:
        out["action_type"] = act
        out["value"] = ""

    return out


# =============================
# Main processing
# =============================
async def process_row(row, sem, args, client, model_name, block_image_dir):
    async with sem:
        blocks_path   = row["blocks_path"]
        target_blocks = row.get("target_blocks", [])
        task_desc     = row["task"]
        # need rerun
        if 'previous_actions_descriptions' in row:
            prev_actions  = row.get("previous_actions_descriptions", []) or []
            if random.random() < 0.02:
                print(f"Using previous_actions_descriptions: {prev_actions}")
        else:
            prev_actions  = row.get("previous_actions", []) or []


        block_num = 0
        response = ""
        parsed: Dict[str, Any] = {"action_type": "", "value": "", "point_2d": [-100, -100], "reason": ""}
        img_size = (0, 0)

        while True:
            img_file = os.path.join(block_image_dir, blocks_path, f"{block_num}.png")
            if not os.path.exists(img_file):
                break

            b64, img_size = encode_image(img_file)
            if not b64:
                break

            # Build official prompt
            system_text, user_text = build_prompt_r1_only(
                agent_family=args.agent_family,
                task_desc=task_desc,
                prev_actions=prev_actions,
            )

            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_text or ""}]},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
                    {"type": "text", "text": user_text}
                ]}
            ]

            response = await get_llm_response(
                client,
                model_name,
                messages,
                temperature=args.temperature,
                max_completion_tokens=args.max_completion_tokens
            )
            if response is None:
                break

            if args.agent_family == "gui_r1":
                parsed = parse_guir1_action(response)
            else:
                parsed = parse_uir1_action(response)

            # print(response, parsed)
            # breakpoint()

            # Decide if go next block:
            max_target = max(map(int, target_blocks)) if target_blocks else -1
            next_block_exists = os.path.exists(os.path.join(block_image_dir, blocks_path, f"{block_num + 1}.png"))

            act = str(parsed.get("action_type", "")).lower()
            val = str(parsed.get("value", "")).lower()

            # GUI-R1 scroll: value is direction
            # UI-R1 scroll: we default to "down" if missing
            is_scroll_down = (act == "scroll" and ("down" in val or val == ""))

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


async def main(args, client, model_name, block_image_dir):
    sem = asyncio.Semaphore(args.sem_limit)

    with open(args.input_file, "r") as f:
        rows = [json.loads(line) for line in f]

    tasks = [
        asyncio.create_task(process_row(r, sem, args, client, model_name, block_image_dir))
        for r in rows
    ]

    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        results.append(await coro)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as out:
        for r in results:
            out.write(json.dumps(r) + "\n")

    print(f"Done – saved {len(results)} rows to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Official-prompt runner for GUI-R1 + UI-R1 using OpenAI-compatible vLLM endpoint."
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model name passed to /v1/chat/completions")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--blocks", type=str, required=True)

    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--sem_limit", type=int, default=64)
    parser.add_argument("--max_completion_tokens", type=int, default=2048)

    parser.add_argument("--agent_family", type=str, default="gui_r1",
                        choices=["gui_r1", "ui_r1"])

    args = parser.parse_args()

    block_image_dir = args.blocks
    model_name = args.model

    client = AsyncOpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key="token-abc123",
    )

    asyncio.run(main(args, client, model_name, block_image_dir))
