import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import os
import io
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import os
import json
from tqdm import tqdm
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset, DataLoader
import argparse
import re
from datasets import load_dataset
from datasets import Dataset as hf_dataset
from PIL import Image
from io import BytesIO
import sys, time, base64, signal, atexit, threading
import requests
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from queue import Queue as ThreadQueue
import random
import itertools
from openai import OpenAI, AsyncOpenAI
import asyncio, httpx
import datasets
import polars as pl


def extract_is_action_match(text: str):
    """
    Extracts the boolean value of `"is_action_match"` from a string like:
    
    <answer>
    {"is_action_match": false}
    </answer>
    """
    m = re.search(r'"is_action_match"\s*:\s*(true|false)', text)
    if not m:
        return None  # or raise an error if you prefer
    return m.group(1) == "true"


def extract_bbox(text: str):
    if '<answer>' in text and '</answer>' in text:
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        answer = text.split('<answer>')[-1].split('</answer>')[0].strip()
    elif '</think>' in text:
        answer = text.split('</think>')[-1].strip()
    else:
        answer = text
    
    # load as json
    try:
        action_info = json.loads(answer)
        return action_info
    except:        
        action_info = {}
        box_2d_match = re.search(r'"box_2d":\s*\[\s*\[([^\]]*?)\]\s*\]', answer, re.DOTALL)
        # 可能只能match到一个 [,,,]，没有外层的[]
        box_2d_match_2 = re.search(r'"box_2d":\s*\[([^\]]*?)\]', answer, re.DOTALL)
        if not box_2d_match and not box_2d_match_2:
            print("Failed to extract bbox from answer:", answer)
            return {}
        if box_2d_match:
            box_2d_str = box_2d_match.group(1)
            box_2d_values = [float(x.strip()) for x in box_2d_str.split(',')]
            action_info['box_2d'] = box_2d_values
        elif box_2d_match_2:
            box_2d_str = box_2d_match_2.group(1)
            box_2d_values = [float(x.strip()) for x in box_2d_str.strip('[]').split(',')]
            action_info['box_2d'] = box_2d_values
        
        print(answer, action_info)
        return action_info['box_2d'] if 'box_2d' in action_info else [-1000, -1000, -1000, -1000]

def _preprocess_to_b64(processor, sample):
    # load image from bytes
    img = Image.open(BytesIO(sample['image_bytes'])).convert('RGB')
    w, h = img.size
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    out = dict(sample)
    out.pop('image_bytes', None)
    out["image_b64"] = b64
    out["orig_w"] = w
    out["orig_h"] = h
    out["scale"]=[1.0, 1.0]
    out["image_size"]=[w, h]
    return out



def _build_payload(sample, model_path):
    messages = sample['message']
    bs64 = sample['image_b64']
    # find and replace image paths with base64 data
    for i, msg in enumerate(messages):
        if 'content' in msg:
            new_content = []
            for c in msg['content']:
                if c['type'] == 'image':
                    new_content.append(
                        {"type":"image_url",
                        "image_url":{"url":f"data:image/png;base64,{bs64}",
                                    "detail":"high"}},
                    )
                else:
                    new_content.append(c)
            messages[i]['content'] = new_content

    return {
        "model": model_path,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 0.98,
        "repetition_penalty": 1.0,
        "max_tokens": 2048,
    }


async def _send_one(client, url, headers, payload, meta):
    print(f"Sending to {url} ...")
    r = await client.post(url, headers=headers, content=json.dumps(payload),
                            timeout=httpx.Timeout(connect=10.0, read=300.0, write=300.0, pool=10.0),)
    if r.status_code != 200:
        # 打印 server 提示，直指问题
        print(f"[{url}] {r.status_code} {r.reason_phrase} -> {r.text[:500]}")
        r.raise_for_status()

    data_json = r.json()
    text = data_json["choices"][0]["message"]["content"]
    out = dict(meta)
    out["pred"] = text
    # bbox = extract_bbox(text)
    # print("Extracted bbox:", bbox)
    # return bbox
    print('output text:', text)
    return text

    


async def pipeline_send(data_iter, model_path, chat_urls,  
                        headers, cpu_workers=16, queue_size=2048,
                        async_concurrency=256, pbar=None,
                        results=[], max_pixels=2646000):
    """
    data_iter: 可迭代的原始样本
    cpu_workers: Stage1 进程数
    queue_size: 队列容量（防止内存暴涨，提供背压）
    async_concurrency: Stage2 并发度
    """
    limits = httpx.Limits(max_keepalive_connections=512, max_connections=512)
    q = asyncio.Queue(maxsize=queue_size)
    total = 0
    processed = 0
    processed_lock = asyncio.Lock()

    inflight_total = 0
    inflight_by_url = {u: 0 for u in chat_urls}
    inflight_lock = asyncio.Lock()
    async def inc(url):
        nonlocal inflight_total
        async with inflight_lock:
            inflight_total += 1
            inflight_by_url[url] += 1

    async def dec(url):
        nonlocal inflight_total
        async with inflight_lock:
            inflight_total -= 1
            inflight_by_url[url] -= 1


    async def heartbeat():
        while True:
            await asyncio.sleep(2.0)
            by_url_str = " ".join([f"{k.split(':')[-1]}={v}" for k, v in inflight_by_url.items()])
            print(f"[HB] q={q.qsize()}/{q.maxsize} "
                f"inflight_total={inflight_total} {by_url_str} consumers={num_consumers}")

    # 1) 生产者：多进程预处理 -> 放入队列
    def produce_batched(executor, it, batch=64):
        processor = AutoProcessor.from_pretrained(model_path)
        # processor.image_processor.max_pixels = max_pixels
        buf = []
        for s in it:
            buf.append(s)
            if len(buf) >= batch:
                yield [executor.submit(_preprocess_to_b64, processor, x) for x in buf]
                buf.clear()
        if buf:
            yield [executor.submit(_preprocess_to_b64, processor, x) for x in buf]

    # 2) 消费者：从队列取 -> 选择URL -> 发请求 -> 写结果
    async def consumer_loop(client: httpx.AsyncClient, url: str, cid: int):
        print(f"[CONS#{cid}] start for {url}")
        nonlocal processed
        while True:
            item = await q.get()
            if item is None:
                q.task_done()
                break
            payload, meta = item
            try:
                await inc(url)          
                bbox = await _send_one(client, url, headers, payload, meta)
                results.append((bbox, meta))
            except Exception as e:
                print(f"[CONS#{cid}@{url}] send error:", repr(e))
            finally:
                await dec(url)   
                q.task_done()
                if pbar is not None:
                    async with processed_lock:
                        processed += 1
                        # 这里只在每 100 个时更新一次进度条
                        if processed % 100 == 0:
                            pbar.update(100)


    # 3) 并发运行
    default_timeout = httpx.Timeout(connect=20.0, read=300.0, write=300.0, pool=20.0)
    limits = httpx.Limits(max_keepalive_connections=128, max_connections=128)

    # with ProcessPoolExecutor(max_workers=cpu_workers) as ex:
    with ThreadPoolExecutor(max_workers=cpu_workers) as ex:
        # 为每个 URL 创建一个 client
        clients = [(url, httpx.AsyncClient(limits=limits, timeout=default_timeout))
                for url in chat_urls]

        # 每个 URL 分配相同数量的消费者
        per_url_cons = max(1, async_concurrency // max(1, len(clients)))
        consumers = []
        for url, client in clients:
            for i in range(per_url_cons):
                consumers.append(asyncio.create_task(consumer_loop(client, url, i)))

        # 根据真实消费者数量，发送相同数量的哨兵
        num_consumers = len(consumers)

        async def producer_loop(executor):
            produced = 0
            for fut_batch in produce_batched(executor, data_iter, batch=16):  # 可把 64 -> 128
                for fut in as_completed(fut_batch):
                    try:
                        s = fut.result()
                    except Exception as e:
                        print("[PROD] preprocess error:", repr(e))
                        continue
                    payload = _build_payload(s, model_path)
                    s.pop('message', None)  # 减小队列压力
                    await q.put((payload, s))
                    produced += 1
            # 投递哨兵（和消费者数量一致）
            for _ in range(num_consumers):
                await q.put(None)
            print(f"[PROD] done, queued {produced} items + {num_consumers} sentinels")

        hb = asyncio.create_task(heartbeat())

        try:
            prod = asyncio.create_task(producer_loop(ex))
            await asyncio.gather(prod)   # 只等生产者完成投递
            await q.join()               # 等队列清空
            # 关闭写入队列
        finally:
            hb.cancel()
            # 等所有消费者退出后再关闭各自的 client
            await asyncio.gather(*consumers, return_exceptions=True)
            await asyncio.gather(*(c.aclose() for _, c in clients), return_exceptions=True)
    
    if pbar is not None:
        # 把 pbar.n 调整到 processed，再 refresh 一次
        pbar.n = processed
        pbar.refresh()




def main(args):
    MODEL_PATH = args.model_path
    DATA_PATH  = args.data_path
    IMAGE_DIR  = args.image_dir

    num_gpus   = args.num_gpus
    start_port = args.start_port


    # 收集所有可用的 chat_url
    base_urls = ['http://127.0.0.1:%d' % (start_port + i) for i in range(num_gpus)]
    chat_urls = [bu + "/v1/chat/completions" for bu in base_urls]
    headers = {"Content-Type": "application/json"}

    data_list = json.load(open(DATA_PATH, 'r'))
    if args.end_index > args.start_index:
        data_list = data_list[args.start_index:args.end_index]
    print(f"Total samples: {len(data_list)}")

    # 根据评估模式配置输出格式
    if args.eval_mode == "coord":
        # 坐标评估：模型返回 box_2d（0-1000 范围的归一化坐标）
        output_format = '''Your output should follow this format with range 0-1000 for coordinates:
<answer>
{"box_2d": [xmin, ymin, xmax, ymax]}
</answer>'''
        system_prompt = (
            "You are a GUI agent. You are given an instruction and a screenshot of the screen. "
            "Based on these information, you must output the bounding box of the target element in correct JSON format. "
            "The coordinates should be normalized to the range [0, 1000] for both x and y axes."
        )
    else:
        # 动作匹配评估：模型只需要判断 action 是否与 instruction 匹配
        output_format = '''Your output should follow this format:
Thought: <your reasoning about whether the action matches the instruction>
<answer>
{"is_action_match": true/false}
</answer>'''
        system_prompt = (
            "You are a GUI agent. You are given an instruction and a screenshot of the screen. "
            "Based on these information, you must output whether the action matches the instruction in correct JSON format. "
            "You do not need to consider the achievability of the action, only focus on whether the action type and instruction match."
        )

    # process dataset to extract question and answer
    new_data = []
    require_process_data = []
    for d in data_list:
        action = d['action']
        action_type = action['action_type'].lower()
        if action_type in ['click', 'long_press', 'type_text']:
            # no need to process
            new_data.append(d)
            continue
        # handle bounding box 
        if len(d['step_instruction']):
            instruction = d['step_instruction']
        else:
            continue

        img_name = d['screenshot']
        img_path = os.path.join(IMAGE_DIR, img_name)
        with open(img_path, 'rb') as img_f:
            img_bytes = img_f.read()
        
        
        conversations = []
        conversations.append({'role':'system', 'value': ''})
        if args.eval_mode == "coord":
            # 让模型预测目标元素的 bounding box
            conversations.append({'role':'user', 'value': '<image>Please generate the bounding box according to the UI screenshot and instruction:'})
        else:
            # 让模型判断 action 是否与 instruction 匹配
            conversations.append({'role':'user', 'value': '<image>Please evaluate whether the action match the instruction in GUI agent, especially the action type. Note that "terminate" action matches  {"action_type": "status", "goal_status": "successful"}. OpenAPP action can open any APP without the requirement to be visible:'})

        conversations[0]['value'] = system_prompt + '\n\n'
        conversations[1]['value'] += f'Instruction: {instruction}, Action: {action}' + '\n\n' + output_format

        new_conversations = []
        for conv in conversations:
            tmp_value = conv.pop('value')
            if '<image>' not in tmp_value:
                conv['content'] = [{"type": "text", "text": tmp_value}]
                new_conversations.append(conv)
            else:
                tmp_value = tmp_value.replace("<image>", "")
                conv["content"] = [{'type': "image", "image": ''}] + [{"type": "text", "text": tmp_value}]
                new_conversations.append(conv)
    
        require_process_data.append({
            'message': new_conversations,
            'orig_data': d,
            'image_bytes': img_bytes,
        })     

    print(f"Total samples require bbox extraction: {len(require_process_data)}")
    pbar = tqdm(total=len(data_list), desc="pipeline(chat.completions)")
    results = []
    
    try:
        asyncio.run(pipeline_send(
            data_iter=require_process_data,                 # 源数据可迭代
            model_path=args.model_path,
            chat_urls=chat_urls,
            headers=headers,
            cpu_workers=args.cpu_workers,           # e.g., 16/32
            queue_size=args.queue_size,             # e.g., 1024
            async_concurrency=args.async_concurrency,  # e.g., 256
            results=results,
            max_pixels=args.max_pixels,
            pbar=pbar,
        ))
    finally:
        pbar.close()

    # 根据评估模式处理模型输出
    for res, meta in results:
        orig_data = meta['orig_data']

        if args.eval_mode == "coord":
            # 使用 bbox 判断坐标是否落在预测框中
            bbox = extract_bbox(res)
            if isinstance(bbox, dict):
                gt_bbox = bbox['box_2d'] if 'box_2d' in bbox else [-1000, -1000, -1000, -1000]
            elif isinstance(bbox, list) and len(bbox) == 4:
                gt_bbox = bbox
            else:
                gt_bbox = [-1000, -1000, -1000, -1000]

            if 'x' in orig_data['action'] and 'y' in orig_data['action']:
                orig_point = [orig_data['action']['x'], orig_data['action']['y']]
            else:
                continue

            x1, y1, x2, y2 = gt_bbox
            width, height = orig_data['screenshot_width'], orig_data['screenshot_height']
            x1 = x1 / 1000 * width
            y1 = y1 / 1000 * height
            x2 = x2 / 1000 * width
            y2 = y2 / 1000 * height

            if x1 <= orig_point[0] <= x2 and y1 <= orig_point[1] <= y2:
                new_data.append(orig_data)
            else:
                print('episode_id:', orig_data['episode_id'], 'step:', orig_data['step'])
        else:
            # 使用 is_action_match 判断动作是否与指令匹配
            is_match = extract_is_action_match(res)
            if is_match:
                new_data.append(orig_data)

    print(f"Total processed results: {len(results)}")
    print(f"Processed {len(new_data)} samples.")

    # save new_data to json file
    output_file = DATA_PATH.replace('.json', '_filtered_bbox_matchinstruction.json')
    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=4)
    print(f"Saved filtered data to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen3-VL-32B-Instruct')
    parser.add_argument('--data_path', type=str, required=True, help="Path to the AndroidControl data JSON file")
    parser.add_argument('--image_dir', type=str, required=True, help="Path to the AndroidControl screenshot images directory")
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--start_port', type=int, default=8000)
    parser.add_argument('--max_pixels', type=int, default=2646000)
    parser.add_argument('--cpu_workers', type=int, default=32)
    parser.add_argument('--async_concurrency', type=int, default=32)
    parser.add_argument('--queue_size', type=int, default=128)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=-1)

    args = parser.parse_args()
    main(args)

