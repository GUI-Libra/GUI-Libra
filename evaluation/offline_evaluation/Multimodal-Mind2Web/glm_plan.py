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

SEM_LIMIT = 100
 
system_prompt = '''Imagine that you are imitating humans doing web navigation for a task step by step.
At each stage, you can see the webpage like humans by a screenshot and know the previous actions before the current step decided by yourself through recorded history.
You need to decide on the first following action to take.
You can click an element with the mouse, select an option, type text with the keyboard, or scroll down.(For your understanding, they are like the click(), select_option(), type() and mouse.wheel() functions in playwright respectively.)'''



question_description = '''You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## The Instruction is: {}

## Previous actions are: {}

## Response Format
Ensure your answer strictly follows the format and requirements provided below, and is clear and precise.:

<think>Output your thought process here step by step.</think>
<answer>
{{
"action_description": "Describe the current step action with a clear instruction",
"action_type": "The type of action you want to take, choosing from CLICK, TYPE, SELECT, or SCROLL DOWN. Before TYPE, you musch first CLICK the input field. The 'Answer' action is used for the last step.",
"action_target": "Provide a description of the element you want to operate.",  (If ACTION == SCROLL DOWN, this field should be None.)
                        It should include the element's identity, type (button, input field, dropdown menu, tab, etc.), and text on it (if have).
                        Ensure your description is both concise and complete, covering all the necessary information and less than 30 words.
                        If you find identical elements, specify its location and details to differentiate it from others. If you already clicked the input field in last step, you can leave this field as 'None'.",
"value": "Provide additional input based on ACTION. If ACTION == TYPE, specify the text to be typed. If ACTION == SELECT, specify the option to be chosen. Otherwise, write 'None'."
}}
</answer>
'''


question_description_no_reasoning = '''You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## The Instruction is: {}

## Previous actions are: {}

## Response Format
Ensure your answer strictly follows the format below (<answer>...</answer>) and requirements provided below, and is clear and precise:

<answer>
{{
"action_description": "Describe the current step action with a clear instruction",
"action_type": "The type of action you want to take, choosing from CLICK, TYPE, SELECT, or SCROLL DOWN. Before TYPE, you musch first CLICK the input field. The 'Answer' action is used for the last step.",
"action_target": "Provide a description of the element you want to operate.",  (If ACTION == SCROLL DOWN, this field should be None.)
                        It should include the element's identity, type (button, input field, dropdown menu, tab, etc.), and text on it (if have).
                        Ensure your description is both concise and complete, covering all the necessary information and less than 30 words.
                        If you find identical elements, specify its location and details to differentiate it from others. If you already clicked the input field in last step, you can leave this field as 'None'.",
"value": "Provide additional input based on ACTION. If ACTION == TYPE, specify the text to be typed. If ACTION == SELECT, specify the option to be chosen. Otherwise, write 'None'."
}}
</answer>
'''


action_format = ''
element_format = ''


value_format = ''

def encode_image(path: str) -> str:
    from PIL import Image
    import io, base64
    with Image.open(path) as im:
        buf = io.BytesIO()
        im.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()


async def get_gpt_response(client, model, messages, temperature=0.5):
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=1000,
        )
        return response.choices[0].message.content
    except openai.APIError as e:
        print(f"Request failed: {e}")
        return None


async def process_row(row, sem, args):
    async with sem:                           # concurrency limiter
        annotation_id = row['annotation_id']
        action_uid    = row['action_uid']
        blocks_path   = row['blocks_path']
        target_blocks = row['target_blocks']
        task_desc     = row['task']
        prev_actions  = row.get('previous_actions', []) or []

        block_num = 0
        response, action, value, element_desc = "", "", "", ""

        while True:
            img_file = os.path.join(block_image_dir, blocks_path,
                                    f"{block_num}.png")
            if not os.path.exists(img_file):
                break

            b64 = encode_image(img_file)

            prev_txt = "".join(
                f"\nStep {i+1}\n Action: {txt}\n" for i, txt in enumerate(prev_actions)
            )
            query = question_description.format(task_desc, prev_txt)

            messages = [
                {"role":"user", "content":[
                    {"type":"image_url",
                     "image_url":{"url":f"data:image/jpeg;base64,{b64}",
                                  "detail":"high"}},
                    {"type":"text", "text":query}
                ]}
            ]

            response = await get_gpt_response(client, model_name, messages, temperature=args.temperature)

            # print(response)
            # parse (same regexes you already had) -------------------------
            import re
            action_match = re.search(r"action_type\":\s*\"(.*?)\"action_target\":", response, re.DOTALL)
            action = action_match.group(1).strip().strip(',').strip('"') if action_match else ""

            # element_match = re.search(r"ELEMENT:\s*(.*?)\s*VALUE:", response, re.DOTALL)
            element_match = re.search(r"action_target\":\s*\"(.*?)\"value\":", response, re.DOTALL)
            element_description = element_match.group(1).strip().strip(',').strip('"') if element_match else ""

            # value_match = re.search(r"VALUE:\s*(.*?)$", response, re.DOTALL)
            value_match = re.search(r"value\":\s*\"(.*?)\"", response, re.DOTALL)
            value = value_match.group(1).strip().strip(',').strip('"') if value_match else ""


            max_target = max(map(int, target_blocks)) if target_blocks else -1
            next_block_exists = os.path.exists(
                os.path.join(block_image_dir, blocks_path, f"{block_num+1}.png"))

            if action != "SCROLL DOWN" or block_num > max_target or not next_block_exists:
                break
            block_num += 1

        row.update({
            "ans_block": block_num,
            "gpt_action": action,
            "gpt_value": value,
            "description": element_desc,
            "response": response
        })
        return row


async def main():
    sem = asyncio.Semaphore(SEM_LIMIT)

    # load rows synchronously (small file)
    with open(args.input_file) as f:
        rows = [json.loads(line) for line in f]

    tasks = [asyncio.create_task(process_row(r, sem, args)) for r in rows]
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
    parser.add_argument("--no_reasoning", type=float, default=0.0, help="If set, no reasoning will be performed")
    parser.add_argument("--port", type=int, default=8000, help="Port for the vLLM server")
    args = parser.parse_args()

    input_file_path = args.input_file
    output_file_path = args.output_file
    block_image_dir = args.blocks
    model_name = args.model

    client = AsyncOpenAI(
        base_url=f"http://localhost:{args.port}/v1",  # vLLM service address
        api_key="token-abc123",  # Must match the --api-key used in vLLM serve
    )

    if args.no_reasoning:
        question_description = question_description_no_reasoning
        
    
    dir_path = os.path.dirname(output_file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")
    else:
        print(f"Directory already exists: {dir_path}")

    asyncio.run(main())
