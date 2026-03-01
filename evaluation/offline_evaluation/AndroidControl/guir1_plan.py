# -*- coding: utf-8 -*-
import argparse
import json
import io
import os
import re
import base64
import asyncio
import ast
import random
from typing import Any, Dict, Tuple, Optional, List

from PIL import Image
import openai
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import aiofiles


# =============================
# Utils: encode image
# =============================
def encode_image(path: str) -> Tuple[Optional[str], Tuple[int, int]]:
    try:
        with Image.open(path) as im:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode(), im.size  # (w, h)
    except Exception as e:
        print(f"[encode_image] Error encoding image {path}: {e}")
        return None, (0, 0)


# =============================
# Official Prompts (AndroidControl)
# =============================

def build_guir1_prompt(task_text: str, history_text: str) -> str:
    """
    GUI-R1 official prompt (Android)
    https://github.com/ritzz-ai/GUI-R1/blob/main/guir1/inference/inference_vllm_android.py
    """
    text = (
        f"You are GUI-R1, a reasoning GUI Agent Assistant. In this UI screenshot <image>, "
        f"I want you to continue executing the command '{task_text}', with the action history being '{history_text}'.\n"
        "Please provide the action to perform (enumerate from ['wait', 'long_press', 'click', 'press_back', 'type', 'open_app', 'scroll', 'complete']), "
        "the point where the cursor is moved to (integer) if a click is performed, and any input text required to complete the action.\n"
        "Output the thinking process in <think> </think> tags, and the final answer in <answer> </answer> tags as follows:\n"
        "<think> ... </think> <answer>[{'action': enum['wait', 'long_press', 'click', 'press_back', 'type', 'open_app', 'scroll', 'complete'], "
        "'point': [x, y], 'input_text': 'no input text [default]'}]</answer>\n"
        "Note:\n specific input text (no default) is necessary for actions enum['type', 'open_app', 'scroll'] \n Example:\n"
        "[{'action': enum['wait', 'press_back'], 'point': [-100, -100], 'input_text': 'no input text'}]\n"
        "[{'action': enum['click', 'long_press'], 'point': [123, 300], 'input_text': 'no input text'}]\n"
        "[{'action': enum['type'], 'point': [-100, -100], 'input_text': 'shanghai shopping mall'}]\n"
        "[{'action': enum['open_app'], 'point': [-100, -100], 'input_text': 'outlook'}]\n"
        "[{'action': enum['scroll'], 'point': [-100, -100], 'input_text': enum['up', 'left', 'right', 'down']}]"
    )
    # Official inference adds this line:
    return "<image>\n" + text


def build_uir1_prompt(task_text: str) -> str:
    """
    UI-R1 official prompt (AndroidControl)
    https://github.com/lll6gg/UI-R1/blob/main/evaluation/test_androidcontrol.py
    NOTE: original snippet has a few quote typos; we keep it "faithful but valid" in practice.
    """
    question_template = (
        f"In this UI screenshot, I want to perform the command '{task_text}'.\n"
        "Please provide the action to perform (enumerate in ['click', 'open_app', 'scroll', 'navigate_back', 'input_text']) "
        "and the coordinate where the cursor is moved to (integer) if click is performed.\n"
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
        "The output answer format should be as follows:\n"
        "<think> ... </think> <answer>[{'action': enum['click', 'open_app', 'scroll', 'navigate_back', 'input_text'], "
        "'coordinate': [x, y]}]</answer>\n"
        "Please strictly follow the format."
    )
    return question_template


def build_history_text(prev_actions: List[str], max_keep: int = 8) -> str:
    if not prev_actions:
        return "None"
    prev_actions = [str(x).strip() for x in prev_actions if str(x).strip()]
    return " | ".join(prev_actions[-max_keep:]) if prev_actions else "None"


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
        print(f"[get_llm_response] Request failed: {e}")
        return None


# =============================
# Parsing: <think> + <answer>[{...}]</answer>
# =============================
def extract_think(resp: str) -> str:
    m = re.search(r"<think>(.*?)</think>", resp or "", flags=re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_answer_payload(resp: str) -> str:
    if not resp:
        return ""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", resp, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    # tolerate missing </answer>
    if "<answer>" in resp:
        return resp.split("<answer>", 1)[1].strip()
    return resp.strip()


def extract_first_list_literal(text: str) -> str:
    """
    Extract first python-like list literal [ {...} ]
    """
    if not text:
        return ""
    m = re.search(r"(\[\s*\{.*?\}\s*\])", text, flags=re.DOTALL)
    if m:
        return m.group(1)
    # tolerate a single dict
    m2 = re.search(r"(\{\s*.*?\s*\})", text, flags=re.DOTALL)
    if m2:
        return f"[{m2.group(1)}]"
    return ""


def parse_answer_list(resp: str) -> List[Dict[str, Any]]:
    payload = extract_answer_payload(resp)
    list_str = extract_first_list_literal(payload)
    if not list_str:
        return []

    # best: python literal eval
    try:
        obj = ast.literal_eval(list_str)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    except Exception:
        pass

    # fallback: json after normalization
    try:
        normalized = list_str.replace("None", "null").replace("True", "true").replace("False", "false")
        obj = json.loads(normalized)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    except Exception:
        return []

    return []


def parse_guir1_output(resp: str) -> Dict[str, Any]:
    """
    GUI-R1 expects: [{'action': ..., 'point': [x,y], 'input_text': ...}]
    """
    out = {
        "action_type": "",
        "value": "",
        "point_2d": [-100, -100],
        "reason": extract_think(resp),
        "raw_action": "",
        "raw_input_text": "",
    }

    lst = parse_answer_list(resp)
    if not lst:
        return out

    d = lst[0]
    act = str(d.get("action", "")).strip().lower()
    pt = d.get("point", [-100, -100])
    it = d.get("input_text", "no input text")

    out["raw_action"] = act
    out["raw_input_text"] = "" if it is None else str(it).strip()

    # map to unified fields
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
        out["point_2d"] = [-100, -100]
        out["value"] = out["raw_input_text"]

    elif act == "open_app":
        out["action_type"] = "OpenApp"
        out["point_2d"] = [-100, -100]
        out["value"] = out["raw_input_text"]

    elif act == "scroll":
        out["action_type"] = "Scroll"
        out["point_2d"] = [-100, -100]
        out["value"] = out["raw_input_text"].lower()  # up/down/left/right

    elif act == "press_back":
        out["action_type"] = "NavigateBack"
        out["point_2d"] = [-100, -100]
        out["value"] = "None"

    elif act == "wait":
        out["action_type"] = "Wait"
        out["point_2d"] = [-100, -100]
        out["value"] = "5"

    else:
        out["action_type"] = act
        out["value"] = out["raw_input_text"]

    return out


def parse_uir1_output(resp: str) -> Dict[str, Any]:
    """
    UI-R1 expects: [{'action': ..., 'coordinate': [x,y]}]
    """
    out = {
        "action_type": "",
        "value": "",
        "point_2d": [-100, -100],
        "reason": extract_think(resp),
        "raw_action": "",
    }

    lst = parse_answer_list(resp)
    if not lst:
        return out

    d = lst[0]
    act = str(d.get("action", "")).strip().lower()
    coord = d.get("coordinate", [-100, -100])

    out["raw_action"] = act

    if act == "click":
        out["action_type"] = "Click"
        if isinstance(coord, (list, tuple)) and len(coord) == 2:
            out["point_2d"] = [int(coord[0]), int(coord[1])]
        out["value"] = "None"

    elif act == "open_app":
        out["action_type"] = "OpenApp"
        out["point_2d"] = [-100, -100]
        out["value"] = ""  # template doesn't define app_name field

    elif act == "scroll":
        out["action_type"] = "Scroll"
        out["point_2d"] = [-100, -100]
        out["value"] = "down"  # template doesn't include direction -> default down

    elif act == "navigate_back":
        out["action_type"] = "NavigateBack"
        out["point_2d"] = [-100, -100]
        out["value"] = "None"
    
    elif act == "navigate_home":
        out["action_type"] = "NavigateHome"
        out["point_2d"] = [-100, -100]
        out["value"] = "None"
        
    elif act == "complete":
        out["action_type"] = "Terminate"
        out["point_2d"] = [-100, -100]
        out["value"] = "success"

    elif act == "input_text":
        out["action_type"] = "Write"
        out["point_2d"] = [-100, -100]
        out["value"] = ""  # template doesn't define text field

    else:
        out["action_type"] = act
        out["value"] = "None"

    return out


# =============================
# Per-row processing
# =============================
async def process_row(row: Dict[str, Any], sem: asyncio.Semaphore, args, client, model_name: str):
    async with sem:
        episode_id = row.get("episode_id", row.get("annotation_id", ""))
        high_level_instruction = row.get("goal", "")
        low_level_instruction = row.get("step_instruction", "")

        prev_actions = row.get("previous_actions", []) or []
        instruction = high_level_instruction if args.level == "high" else low_level_instruction

        # screenshot path (AndroidControl)
        img_file = os.path.join(args.screenshot_dir, row["screenshot"])
        b64, img_size = encode_image(img_file)
        if not b64:
            return {
                "episode_id": episode_id,
                "step": row.get("step", -1),
                "instruction": instruction,
                "action_type": "",
                "value": "",
                "point_2d": [-100, -100],
                "reason": "",
                "response": "",
                "error": f"encode failed: {img_file}"
            }

        history_text = build_history_text(prev_actions)

        # Build prompt
        if args.agent_family == "gui_r1":
            system_text = ""
            user_text = build_guir1_prompt(task_text=instruction, history_text=history_text)
        else:
            system_text = ""
            user_text = build_uir1_prompt(task_text=instruction)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
                {"type": "text", "text": user_text},
            ]}
        ]

        response = await get_llm_response(
            client=client,
            model=model_name,
            messages=messages,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
        )

        if response is None:
            return {
                "episode_id": episode_id,
                "step": row.get("step", -1),
                "instruction": instruction,
                "action_type": "",
                "value": "",
                "point_2d": [-100, -100],
                "reason": "",
                "response": "",
                "error": "llm response None"
            }

        # Parse
        if args.agent_family == "gui_r1":
            parsed = parse_guir1_output(response)
        else:
            parsed = parse_uir1_output(response)

        # occasional debug prints
        if random.random() < args.print_ratio:
            print("\n==================== SAMPLE OUTPUT ====================")
            print("instruction:", instruction)
            print("response:", response)
            print("parsed:", parsed)
            print("=======================================================\n")
            # breakpoint()

        result = {
            "episode_id": episode_id,
            "step": row.get("step", -1),
            "level": args.level,
            "instruction": instruction,
            "screenshot": row.get("screenshot", ""),
            "image_width": img_size[0],
            "image_height": img_size[1],

            # unified fields
            "action_type": parsed.get("action_type", ""),
            "value": parsed.get("value", ""),
            "point_2d": parsed.get("point_2d", [-100, -100]),
            "reason": parsed.get("reason", ""),

            # raw model output
            "response": response,

            # raw fields helpful for debugging
            "raw_action": parsed.get("raw_action", ""),
            "raw_input_text": parsed.get("raw_input_text", ""),
        }
        return result


# =============================
# Main
# =============================
async def main(args, client, model_name: str):
    sem = asyncio.Semaphore(args.sem_limit)

    # your input file is JSON list
    with open(args.input_file, "r") as f:
        rows = json.load(f)

    tasks = [asyncio.create_task(process_row(r, sem, args, client, model_name)) for r in rows]
    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        results.append(await coro)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    async with aiofiles.open(args.output_file, "w") as out:
        for r in results:
            await out.write(json.dumps(r) + "\n")

    print(f"Done – saved {len(results)} rows to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AndroidControl Mobile Inference (GUI-R1 / UI-R1) via OpenAI-compatible vLLM endpoint.")
    parser.add_argument("--model", type=str, required=True, help="HF local path or model id served by vLLM")
    parser.add_argument("--input_file", type=str, required=True, help="Path to AndroidControl JSON file (list)")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--screenshot_dir", type=str, required=True, help="Directory containing screenshots")

    parser.add_argument("--level", type=str, default="high", choices=["high", "low"], help="Use high-level goal or low-level step instruction")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--port", type=int, default=8000, help="vLLM port")
    parser.add_argument("--sem_limit", type=int, default=64, help="Concurrency limit")
    parser.add_argument("--max_completion_tokens", type=int, default=2048)

    # NEW: choose GUI-R1 or UI-R1 prompt/parse
    parser.add_argument("--agent_family", type=str, default="gui_r1", choices=["gui_r1", "ui_r1"])

    # debug sampling
    parser.add_argument("--print_ratio", type=float, default=0.02, help="Probability to print a random sample output for debugging")

    args = parser.parse_args()

    model_name = args.model
    client = AsyncOpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key="token-abc123",
    )

    asyncio.run(main(args, client, model_name))
