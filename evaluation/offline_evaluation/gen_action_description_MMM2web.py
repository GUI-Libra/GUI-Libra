import os 
import io
import json 
from PIL import Image, ImageDraw
from tqdm import tqdm
import requests
import base64
import time
import openai
from openai import OpenAI, AsyncOpenAI
import argparse
import copy
import re 
import asyncio


client = AsyncOpenAI(
    base_url=f'http://127.0.0.1:8000/v1',
)

mode = 'domain'
image_dir = 'path/to/blocks_images/cross_{}/'.format(mode)
path = 'path/to/samples/cross_{}_blocks.jsonl'.format(mode)
data = []
with open(path, 'r') as f:
    for line in f:
        data.append(json.loads(line))

data_dict = {}
for item in data:
    if item['annotation_id'] not in data_dict:
        total_steps = item['total_steps'] 
        data_dict[item['annotation_id']] = [None]*total_steps

    data_dict[item['annotation_id']][item['step']] = item


print("Total trajectories:", len(data_dict))
# check each trajectory is complete, i.e., not None in any step
pop_list = []
for annotation_id, trajectory in data_dict.items():
    for step_idx, step in enumerate(trajectory):
        if step is None:
            print(f"Error: annotation_id {annotation_id} is missing step {step_idx+1}")
            pop_list.append(annotation_id)
            break

data_dict = {k: v for k, v in data_dict.items() if k not in pop_list}

print("After filtering, total trajectories:", len(data_dict))
breakpoint()

async def get_response_from_openai(step_data, all_actions, image, model_name='Qwen/Qwen3-VL-32B-Instruct'):
    
    instruction = step_data['task']
    current_action_type = step_data['operation']
    current_action_value = step_data['value']
    step_id = step_data['step']
    messages = []
    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_str}",
                        "detail": "high",
                    },
                },
                {
                    "type": "text",
                    "text": f"""You are given a screenshot of a webpage and a task instruction. You need to describe the current step action in one sentence. If the action has a target element, the element is highlighted with a red bounding box in the image, but you cannot describe the red bounding box itself. Instead, you need to identify the target element based on its original visual appearance, position, and textual information.
The task instruction is: {instruction}.
Current action type: {current_action_type}.
Current action value: {current_action_value}.
Current symbolic action (optional): {all_actions[step_id] if step_id < len(all_actions) - 1 else "N/A"}.
Please provide a natural language description of the action to be performed on the webpage in the current step.
"""
                },
            ],
        }
    ]


    response = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        max_tokens=1024,
    )
    print("Response:", response.choices[0].message.content)
    # breakpoint()
        
    return response.choices[0].message.content


async def process_all_annotations(data_dict, image_dir, model_name='Qwen/Qwen3-VL-32B-Instruct',
                                  max_concurrency: int = 16):
    """
    data_dict: {annotation_id: trajectory}
    trajectory: list of steps
    """

    sem = asyncio.Semaphore(max_concurrency)

    async def handle_step(annotation_id, step_idx, step, all_actions):
        # 同步做图像加载和画框（一般没问题，如果特别重可以放 executor）
        target_blocks = list(step['target_blocks'].keys())
        image_index = target_blocks[0] if len(target_blocks) else 0
        image_path = os.path.join(image_dir, step['blocks_path'], f"{image_index}.png")
        if not os.path.exists(image_path):
            return None

        image = Image.open(image_path).convert("RGB")
        boxes = step['bbox']
        draw = ImageDraw.Draw(image)
        for box in boxes:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x1 + x2, y1 + y2], outline="red", width=3)

        # 控制并发，防止把本地模型打爆
        async with sem:
            response = await get_response_from_openai(
                step,
                all_actions,
                image,
                model_name=model_name,
            )

        # 这里你可以根据自己需要存结果存到data_dict
        data_dict[annotation_id][step_idx]['action_description'] = response

    tasks = []
    for annotation_id, trajectory in data_dict.items():
        all_actions = trajectory[-1]['previous_actions']
        for step_idx, step in enumerate(trajectory):
            tasks.append(
                asyncio.create_task(
                    handle_step(annotation_id, step_idx, step, all_actions)
                )
            )

    # 并发执行所有 step
    await asyncio.gather(*tasks)
    return data_dict



async def main():
    # data_dict, image_dir 准备好
    new_data_dict = await process_all_annotations(data_dict, image_dir)
    # revise previous actions to use natural language descriptions
    pop_list = []
    for annotation_id, trajectory in new_data_dict.items():
        previous_descriptions = []
        for step_idx, step in enumerate(trajectory):
            new_data_dict[annotation_id][step_idx]['previous_actions_descriptions'] = copy.deepcopy(previous_descriptions)
            if 'action_description' not in step:
                print(f"Error: annotation_id {annotation_id} is missing action_description in step {step_idx+1}")
                pop_list.append(annotation_id)
                break
            previous_descriptions.append(step['action_description'])
    new_data_dict = {k: v for k, v in new_data_dict.items() if k not in pop_list}


    # 保存结果
    output_path = path.replace('.jsonl', '_natural.jsonl')
    with open(output_path, 'w') as f:
        for annotation_id, trajectory in new_data_dict.items():
            for step in trajectory:
                f.write(json.dumps(step) + '\n')
    


if __name__ == "__main__":
    asyncio.run(main())