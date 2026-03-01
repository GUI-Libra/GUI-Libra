import json 
import collections
import numpy as np
import os
import re 
from PIL import Image
import string
import argparse
import unicodedata

_PUNC_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)  
_WS_RE   = re.compile(r"\s+", flags=re.UNICODE)

def normalize_text(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFKC", s)   
    s = _PUNC_RE.sub(" ", s)           
    s = _WS_RE.sub(" ", s).strip()        
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


def get_correctness(sample, planner_entry):
    if planner_entry:
        response = planner_entry.get("response", "")
        if '\nAction:' in response:
            action = response.split('\nAction:')[-1].strip()
        elif '\nACTION:' in response:
            action = response.split('\nACTION:')[-1].strip()
        else:
            action = response.strip()
        
        # gpt_value = extract_input_text(action).lower()
        # gpt_action = extract_action(action).lower()
        # gpt_coord, coord_found = extract_coord(action)

        if 'gpt_value' in planner_entry:
            gpt_value = planner_entry['gpt_value'].lower()
        else:
            gpt_value = extract_input_text(action).lower()
            
        if 'gpt_action' in planner_entry:
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
            gpt_value = ''
            pred_action = gpt_action
        else:
            pred_action = f"{gpt_action} {gpt_value}"

    else:
        pred_action = ""


    bboxes = planner_entry.get("bbox", [])
    scale = 1 #planner_entry.get("scale", 1.0)
    x, y = output
    if args.relative_coord:
        # breakpoint()
        if 'image_width' in planner_entry and 'image_height' in planner_entry and planner_entry['image_width'] > 0 and planner_entry['image_height'] > 0:
            width = planner_entry['image_width']
            height = planner_entry['image_height']
        else:
            img_file = os.path.join(args.block_image_dir, 'cross_' + planner_entry['split'], planner_entry['blocks_path'], "0.png")
            if not os.path.exists(img_file):
                return None, None
            image = Image.open(img_file)
            width, height = image.size
            
        x = x / 1000 * width
        y = y / 1000 * height
        output = (x, y)

    correct, coords = is_output_inside_bbox(bboxes, output, scale)
    # print(output, planner_entry['bbox'], correct)

    current_action = (sample["operation"].lower(), sample["value"].lower())
    f1_score = calculate_f1(pred_action.lower(), current_action[0]+" "+current_action[1])
    return correct, f1_score 
    

def get_metrics_with_prediction(plan_data):
    all_element_acc = []
    all_operation_f1 = []
    all_step_acc = []
    sample_to_website = {}

    new_sample_data = []

    plan_data_dict = {}
    for data in plan_data:
        for sample in data:
            annotation_id = sample['annotation_id']
            action_uid = sample['action_uid']
            sample_id = f"{annotation_id}_{action_uid}"
            if sample_id not in plan_data_dict:
                plan_data_dict[sample_id] = [sample]
            else:
                plan_data_dict[sample_id].append(sample)

    total_steps = {}
    for samples in plan_data_dict.values():
        for s in samples:
            aid = s.get("annotation_id")
            ts = s.get("total_steps", None)
            if aid is None or ts is None:
                continue
            total_steps[aid] = max(total_steps.get(aid, 0), ts)

    for entry_data in plan_data_dict:
        correct_res = []
        f1_score_res = []
        step_res = []

        for sample in plan_data_dict[entry_data]:
            annotation_id = sample['annotation_id']
            action_uid = sample['action_uid']
            sample_id = f"{annotation_id}_{action_uid}"
            
            sample_to_website[annotation_id] = sample["website"]
            
            # Get planner data
            planner_entry = sample
            correct, f1_score = get_correctness(sample, planner_entry)
            if correct is None or f1_score is None:
                continue
            correct_res.append(correct)
            f1_score_res.append(f1_score)
            step_res.append(1 if (correct and f1_score == 1) else 0)  

        if len(correct_res) == 0 or len(f1_score_res) == 0 or len(step_res) == 0:
            continue
        all_element_acc.append([1 if any(correct_res) else 0, annotation_id])
        all_operation_f1.append([max(f1_score_res), annotation_id])
        step_best = 1 if any(step_res) else 0
        all_step_acc.append([step_best, annotation_id])
    

    current_steps = collections.defaultdict(int)
    for _, annotation_id in all_element_acc:
        current_steps[annotation_id] += 1
    
    total_added = 0
    for annotation_id, steps in total_steps.items():
        while current_steps[annotation_id] < steps:
            all_element_acc.append([0, annotation_id])
            all_operation_f1.append([0, annotation_id])
            all_step_acc.append([0, annotation_id])
            current_steps[annotation_id] += 1
            total_added += 1
    print(f"Total padded steps: {total_added}")
            
    
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
    }

# Load data
parser = argparse.ArgumentParser(description="Calculate metrics for Mind2Web data")
parser.add_argument("--plan_file", type=str, required=True, help="Path to plan JSONL file")
parser.add_argument("--block_image_dir", type=str, required=True, help="Base directory for block images")
parser.add_argument("--file_num", type=int, default=1, help="Number of answer files to process (default: 1)")
parser.add_argument("--relative_coord", action='store_true', default=False, help="Whether to use relative coordinates")


args = parser.parse_args()
num_files = args.file_num

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

# Load data
plan_file_list = []
for i in range(0, num_files):
    temp_plan_file = args.plan_file.replace('_0_', f'_{i}_')
    if not os.path.exists(temp_plan_file):
        print(f"Plan file {temp_plan_file} does not exist, skipping.")
        continue
    plan_file_list.append(temp_plan_file)

plan_data = []
for plan_file in plan_file_list:
    plan_data.append(load_jsonl(plan_file))

# Calculate metrics
metrics = get_metrics_with_prediction(plan_data)

# Print results
print("Metrics:")
for key, value in metrics.items():
    if not isinstance(value, dict):
        print(f"{key}: {value*100:.2f}%")
