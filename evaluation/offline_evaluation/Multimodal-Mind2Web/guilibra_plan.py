# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import json
import io
from PIL import Image
from tqdm import tqdm
import os
import re
import base64
import requests
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor
import torch
from qwen_vl_utils import process_vision_info
from openai import OpenAI
import openai
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm  
import aiofiles
import random

SEM_LIMIT = 100

system_prompt = """You are a GUI agent. You are given a task and a screenshot of the screen. You need to choose actions from the the following list:
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
"""


question_description = '''Please generate the next move according to the UI screenshot {}, instruction and previous actions.\n\nInstruction: {}\n\nInteraction History: {}\n'''


def encode_image(path: str) -> str:
    from PIL import Image
    import io, base64
    try:
        with Image.open(path) as im:
            buf = io.BytesIO()
            im.save(buf, format="JPEG")
            return base64.b64encode(buf.getvalue()).decode(), im.size
    except Exception as e:
        print(f"Error encoding image {path}: {e}")
        return None, (0, 0)

async def get_gpt_response(client, model, messages, temperature=0.5):
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


async def process_row(row, sem):
    async with sem:                           # concurrency limiter
        annotation_id = row['annotation_id']
        action_uid    = row['action_uid']
        blocks_path   = row['blocks_path']
        target_blocks = row['target_blocks']
        task_desc     = row['task']
        ##### added just now
        if 'previous_actions_descriptions' in row:
            prev_actions  = row.get("previous_actions_descriptions", []) or []
            if random.random() < 0.02:
                print(f"Using previous_actions_descriptions: {prev_actions}")
        else:
            prev_actions  = row.get("previous_actions", []) or []


        block_num = 0
        response, action, value, element_desc, reason = "", "", "", "", ""

        img_size = (0, 0)
        while True:
            img_file = os.path.join(block_image_dir, blocks_path,
                                    f"{block_num}.png")
            if not os.path.exists(img_file):
                break

            b64, img_size = encode_image(img_file)
            if not b64:
                print(f"Failed to encode image: {img_file}")
                continue

            prev_txt = "".join(
                f"\nStep {i+1}\n Action: {txt}\n" for i, txt in enumerate(prev_actions)
            )
            img_size_string = '(original image size {}x{})'.format(img_size[0], img_size[1])
            query = question_description.format(img_size_string, task_desc, prev_txt)
            if args.add_template:
                if args.reasoning:
                    if 'qwen3' in args.model.lower() or 'libra-4b' in args.model.lower() or 'libra-8b' in args.model.lower():
                        query = query + '\n' + '''The response should be structured in the following format:
<thinking>Your step-by-step thought process here...</thinking>
<answer>
{
  "action_type": "the type of action to perform, e.g., Click, Write, Scroll, Answer, etc. Please follow the system prompt for available actions.",
  "action_target": "the description of the target of the action, such as the color, text, or position on the screen of the UI element to interact with",
  "value": "the input text or direction ('up', 'down', 'left', 'right') for the 'scroll' action, if applicable; otherwise, use 'None'",
  "point_2d": [x, y] # the coordinates on the screen where the action is to be performed; if not applicable, use [-100, -100]
}
</answer>'''
                    else:
                        query = query + '\n' + '''The response should be structured in the following format:
<think> Your step-by-step thought process here... </think>
<answer>
{
  "action_description": "the description of the action to perform summarized in one sentence",  
  "action_type": "the type of action to perform, e.g., Click, Write, Scroll, Answer, etc. Please follow the system prompt for available actions.",
  "action_target": "the description of the target of the action, such as the color, text, or position on the screen of the UI element to interact with",
  "value": "the input text or direction ('up', 'down', 'left', 'right') for the 'scroll' action, if applicable; otherwise, use 'None'",
  "point_2d": [x, y] # output the coordinates on the screen where the action is to be performed; if not applicable, use [-100, -100] or None
}
</answer>'''
                else:
                    query = query + '\n' + '''The response should be structured in the following format:
<answer>
{
  "action_description": "the description of the action to perform summarized in one sentence",  
  "action_type": "the type of action to perform, e.g., Click, Write, Scroll, Answer, etc. Please follow the system prompt for available actions.",
  "action_target": "the description of the target of the action, such as the color, text, or position on the screen of the UI element to interact with",
  "value": "the input text or direction ('up', 'down', 'left', 'right') for the 'scroll' action, if applicable; otherwise, use 'None'",
  "point_2d": [x, y] # output the coordinates on the screen where the action is to be performed; if not applicable, use [-100, -100] or None
}
</answer>'''


            messages = [
                {"role":"system",
                 "content":[{"type":"text",
                             "text": system_prompt}]},
                {"role":"user", "content":[
                    {"type":"image_url",
                     "image_url":{"url":f"data:image/png;base64,{b64}",
                                  "detail":"high"}},
                    {"type":"text", "text":query}
                ]}
            ]

            response = await get_gpt_response(client, model_name, messages, temperature=args.temperature)
            if random.random() < 0.02:
                print(response)
            think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
            reason = think_match.group(1).strip() if think_match else ""
            action_match = re.search(r"action_type\":\s*\"(.*?)\"action_target\":", response, re.DOTALL)
            action = action_match.group(1).strip().strip(',').strip('"') if action_match else ""

            element_match = re.search(r"action_target\":\s*\"(.*?)\"value\":", response, re.DOTALL)
            element_desc = element_match.group(1).strip().strip(',').strip('"') if element_match else ""

            value_match = re.search(r"value\":\s*\"(.*?)\"", response, re.DOTALL)
            value = value_match.group(1).strip().strip(',').strip('"') if value_match else ""


            max_target = max(map(int, target_blocks)) if target_blocks else -1
            next_block_exists = os.path.exists(
                os.path.join(block_image_dir, blocks_path, f"{block_num+1}.png"))

            if not (action.lower() == "scroll" and value.lower() == 'down') or block_num > max_target or not next_block_exists:
                break
            block_num += 1

        row.update({
            "ans_block": block_num,
            "gpt_action": action,
            "gpt_value": value,
            "description": element_desc,
            "response": response,
            "reason": reason,
            'image_width': img_size[0],
            'image_height': img_size[1],
        })
        return row


async def main():
    sem = asyncio.Semaphore(SEM_LIMIT)

    # load rows synchronously (small file)
    with open(args.input_file) as f:
        rows = [json.loads(line) for line in f]

    tasks = [asyncio.create_task(process_row(r, sem)) for r in rows]
    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        results.append(await coro)

    async with aiofiles.open(args.output_file, "w") as out:
        for r in results:
            await out.write(json.dumps(r) + "\n")

    print(f"Done – saved {len(results)} rows to {args.output_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process web navigation tasks using GPT-4V.")
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct', help='the name of GPT model to use')
    parser.add_argument("--input_file", type=str, required=True, help="Path to sample(blocks) JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output plan JSONL file")
    parser.add_argument("--blocks", type=str, required=True, help="Directory for block images")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for GPT model")
    parser.add_argument("--port", type=int, default=8000, help="port for serving model")
    parser.add_argument("--add_template", type=int, default=0, help="Whether to add template to the prompt")
    parser.add_argument("--reasoning", type=int, default=1, help="Whether to include reasoning in the prompt")
    parser.add_argument("--sem_limit", type=int, default=64, help="Semaphore limit for concurrency")

    args = parser.parse_args()

    input_file_path = args.input_file
    output_file_path = args.output_file
    block_image_dir = args.blocks
    model_name = args.model

    client = AsyncOpenAI(
        base_url=f"http://localhost:{args.port}/v1",  # vLLM service address
        api_key="token-abc123",  # Must match the --api-key used in vLLM serve
    ) 

    dir_path = os.path.dirname(output_file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")
    else:
        print(f"Directory already exists: {dir_path}")

    asyncio.run(main())
