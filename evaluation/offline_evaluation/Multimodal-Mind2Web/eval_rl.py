import json 
import collections
import numpy as np
import os
import re 
from PIL import Image
import string
import argparse
import unicodedata

_PUNC_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)  # 非字母数字下划线/空白的都当标点
_WS_RE   = re.compile(r"\s+", flags=re.UNICODE)

def normalize_text(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFKC", s)   # 统一全角半角等
    s = _PUNC_RE.sub(" ", s)               # 标点→空格
    s = _WS_RE.sub(" ", s).strip()         # 压缩空白
    return s

def calculate_f1(pred, label):
    pred = set(normalize_text(pred).split())
    label = set(normalize_text(label).split())
    # remove punctuation
    pred = set([x for x in pred if x not in string.punctuation])
    label = set([x for x in label if x not in string.punctuation])
    if len(pred) == 0 and len(label) == 0:
        return 1
    if len(pred) == 0 or len(label) == 0:
        return 0

    tp = len(pred & label)
    fp = len(pred - label)
    fn = len(label - pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def is_output_inside_bbox(bboxes, output, scale):
    output_x, output_y = output
    output_x /= scale
    output_y /= scale

    for bbox in bboxes:
        bbox_x, bbox_y, bbox_width, bbox_height = bbox
        if bbox_x <= output_x <= bbox_x + bbox_width and bbox_y <= output_y <= bbox_y + bbox_height:
            return True, (output_x, output_y)
    return False, (output_x, output_y)

def extract_action(content):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    action_pattern = r'"action_type":\s*"(.*?)"'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        action_match = re.search(action_pattern, content_answer)
        if action_match:
            return action_match.group(1)
    return "no action"

def extract_input_text(content):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    action_pattern = r'"value":\s*"(.*?)"'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        action_match = re.search(action_pattern, content_answer)
        if action_match:
            return action_match.group(1)
    return "no input text"

def extract_coord(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'"point_2d"\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    try:
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            coord_match = re.search(bbox_pattern, content_answer)
            if coord_match:
                coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                return coord, True
        else:
            coord_pattern = r'\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]'
            coord_match = re.search(coord_pattern, content)
            if coord_match:
                coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                return coord, True
        return [0, 0], False
    except:
        return [0, 0], False

def get_metrics_with_prediction(plan_data):
    all_element_acc = []
    all_operation_f1 = []
    all_step_acc = []
    sample_to_website = {}
    new_sample_data = []
    
    # save data to 
    for sample in plan_data:
        annotation_id = sample['annotation_id']
        action_uid = sample['action_uid']
        sample_id = f"{annotation_id}_{action_uid}"
        
        sample_to_website[annotation_id] = sample["website"]
        
        # Get planner data
        planner_entry = next((item for item in plan_data if item['annotation_id'] == annotation_id and item['action_uid'] == action_uid), None)
        if planner_entry:
            response = planner_entry.get("response", "")
            # breakpoint()
            if '\nAction:' in response:
                action = response.split('\nAction:')[-1].strip()
            elif '\nACTION:' in response:
                action = response.split('\nACTION:')[-1].strip()
            else:
                action = response.strip()
            
            
            if 'gpt_value' in planner_entry and planner_entry['gpt_value'].strip() != '':
                gpt_value = planner_entry['gpt_value'].lower()
            else:
                gpt_value = extract_input_text(action).lower()
            if 'gpt_action' in planner_entry and planner_entry['gpt_action'].strip() != '':
                gpt_action = planner_entry['gpt_action'].lower()
            else:
                gpt_action = extract_action(action).lower()

            if 'point_2d' in planner_entry:
                gpt_coord = planner_entry['point_2d']
                coord_found = True
            else:
                gpt_coord, coord_found = extract_coord(action)

            output = gpt_coord
            if gpt_action == 'no action':
                gpt_action = ''

            action_mapping = {
                'write': 'type',
            }
            if gpt_action in action_mapping:
                gpt_action = action_mapping[gpt_action]
            
            if gpt_action in ['click', 'longpress'] or gpt_value == 'none':
                gpt_value = ''
                            
            if gpt_value == "none":
                pred_action = gpt_action
            else:
                pred_action = f"{gpt_action} {gpt_value}"
        else:
            pred_action = ""
            continue # skip if no planner entry found


        bboxes = planner_entry.get("bbox", [])
        scale = planner_entry.get("scale", 1.0)
        x, y = output
        if args.relative_coord:
            if 'image_width' in planner_entry and 'image_height' in planner_entry and planner_entry['image_width'] > 0 and planner_entry['image_height'] > 0:
                width = planner_entry['image_width']
                height = planner_entry['image_height']
            else:
                img_file = os.path.join(args.block_image_dir, 'cross_' + planner_entry['split'], planner_entry['blocks_path'], "0.png")
                if not os.path.exists(img_file):
                    continue
                image = Image.open(img_file)
                width, height = image.size

            x = x / 1000 * width
            y = y / 1000 * height
            output = (x, y)

        correct, coords = is_output_inside_bbox(bboxes, output, scale)
        # print(output, planner_entry['bbox'], correct)
        all_element_acc.append([1 if correct else 0, annotation_id])

        current_action = (sample["operation"].lower(), sample["value"].lower())
        f1_score = calculate_f1(pred_action.lower(), current_action[0]+" "+current_action[1])
        all_operation_f1.append([f1_score, annotation_id])
        all_step_acc.append([1 if (all_operation_f1[-1][0]==1 and all_element_acc[-1][0]==1) else 0, annotation_id])
        # print(f"Pred Action: {pred_action}, Gold Action: {current_action[0]} {current_action[1]}, F1: {f1_score}, Element Acc: {correct}, Step Acc: {all_step_acc[-1][0]}")
        # breakpoint()

    # total_steps = {sample['annotation_id']: sample['total_steps'] for sample in sample_data}
    total_steps = {sample['annotation_id']: sample['total_steps'] for sample in new_sample_data}
    current_steps = collections.defaultdict(int)
    for _, annotation_id in all_element_acc:
        current_steps[annotation_id] += 1
    for annotation_id, steps in total_steps.items():
        while current_steps[annotation_id] < steps:
            all_element_acc.append([0, annotation_id])
            all_operation_f1.append([0, annotation_id])
            all_step_acc.append([0, annotation_id])
            current_steps[annotation_id] += 1
    
    macro_element_acc = collections.defaultdict(list)
    macro_operation_f1 = collections.defaultdict(list)
    macro_step_acc = collections.defaultdict(list)
    for x in all_element_acc:
        macro_element_acc[x[1]].append(x[0])
    for x in all_operation_f1:
        macro_operation_f1[x[1]].append(x[0])
    for x in all_step_acc:
        macro_step_acc[x[1]].append(x[0])
    
    error_ratio = collections.defaultdict(int)
    acc_per_website = collections.defaultdict(list)
    for annotation_id, x in macro_step_acc.items():
        acc_per_website[sample_to_website[annotation_id]].append(np.mean(x))
        error_count = len([y for y in x if y == 0])
        if error_count <= 3:
            error_ratio[error_count] += 1
        else:
            error_ratio[">3"] += 1
    
    acc_per_website = {k: (np.mean(v), len(v)) for k, v in acc_per_website.items()}
    error_ratio = {k: v/len(macro_element_acc) for k, v in error_ratio.items()}
    macro_element_acc = np.mean([np.mean(x) for x in macro_element_acc.values()])
    macro_operation_f1 = np.mean([np.mean(x) for x in macro_operation_f1.values()])
    macro_step_acc = np.mean([np.mean(x) for x in macro_step_acc.values()])

    return {
        "element_acc": np.mean([x[0] for x in all_element_acc]),
        "operation_f1": np.mean([x[0] for x in all_operation_f1]),
        "step_acc": np.mean([x[0] for x in all_step_acc]),
        # "macro_element_acc": macro_element_acc,
        # "macro_operation_f1": macro_operation_f1,
        # "macro_step_acc": macro_step_acc,
        # "error_ratio": error_ratio,
        # "acc_per_website": acc_per_website,
    }

# Load data
parser = argparse.ArgumentParser(description="Calculate metrics for Mind2Web data")
parser.add_argument("--plan_file", type=str, required=True, help="Path to plan JSONL file")
parser.add_argument("--block_image_dir", type=str, required=True, help="Path to block images directory")
parser.add_argument("--relative_coord", type=int, default=0, choices=[0, 1], help="Whether to use relative coordinates (0/1)")

args = parser.parse_args()

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

# Load data
plan_data = load_jsonl(args.plan_file)

# Calculate metrics
metrics = get_metrics_with_prediction(plan_data)

# Print results
print("Metrics:")
for key, value in metrics.items():
    if not isinstance(value, dict):
        print(f"{key}: {value*100:.2f}%")
