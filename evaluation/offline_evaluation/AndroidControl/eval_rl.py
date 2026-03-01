import json
from collections import defaultdict
import argparse
import re
import os

def bounding_box_contains_point(bbox, x, y):
    return bbox['x_min'] <= x <= bbox['x_max'] and bbox['y_min'] <= y <= bbox['y_max']


def extract_action(content):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    action_pattern = r'"action_type":\s*"(.*?)"'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        action_match = re.search(action_pattern, content_answer)
        if action_match:
            return action_match.group(1)
    else:
        action_match = re.search(action_pattern, content)
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
    else:
        action_match = re.search(action_pattern, content)
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

            coord_pattern = r'\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]'
            coord_match = re.search(coord_pattern, content)
            if coord_match:
                coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                return coord, True
        else:
            coord_pattern = r'\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]'
            coord_match = re.search(coord_pattern, content)
            if coord_match:
                coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                return coord, True
        return [0, 0], False
    except:
        # direct match [x, y] format
        coord_pattern = r'\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]'
        coord_match = re.search(coord_pattern, content)
        if coord_match:
            coord = [int(coord_match.group(1)), int(coord_match.group(2))]
            return coord, True
        return [0, 0], False

def find_smallest_bbox_node(x, y, tree):
    """
    Find the smallest bounding box node that contains the given coordinates
    Returns a tuple of (node, bbox) if found, (None, None) if not found
    """
    smallest_node = None
    smallest_bbox = None
    smallest_area = float('inf')
    
    for node in tree:
        if isinstance(node, dict):
            bbox = node['bbox_pixels']
            if bounding_box_contains_point(bbox, x, y):
                area = (bbox['x_max'] - bbox['x_min']) * (bbox['y_max'] - bbox['y_min'])
                if area < smallest_area:
                    smallest_area = area
                    smallest_node = node
                    smallest_bbox = bbox
        elif isinstance(node, list):
            child_node, child_bbox = find_smallest_bbox_node(x, y, node)
            if child_node:
                child_area = (child_bbox['x_max'] - child_bbox['x_min']) * (child_bbox['y_max'] - child_bbox['y_min'])
                if child_area < smallest_area:
                    smallest_area = child_area
                    smallest_node = child_node
                    smallest_bbox = child_bbox
    
    return smallest_node, smallest_bbox

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def process_file(sample_file, plan_file):
    sample_data = load_json(sample_file)
    plan_data = load_jsonl(plan_file)

    sample_index = {(entry['episode_id'], entry['step']): entry for entry in sample_data}
    results = defaultdict(int)

    correct_types = 0
    correct_steps = 0
    total_steps = 0
    grounding_steps = 0
    correct_grounding_steps = 0

    for plan_entry in plan_data:
        episode_id = plan_entry['episode_id']
        step = plan_entry['step']
        
        sample_entry = sample_index.get((episode_id, step))
        if not sample_entry:
            continue
        width, height = sample_entry['screenshot_width'], sample_entry['screenshot_height']
        
        total_steps += 1
        gold_action = sample_entry['action']
        pred_action_type = plan_entry['action_type'].lower()
        pred_action_text = plan_entry['value'].lower()
        if not len(pred_action_type):
            action_match = re.search(r'"action_type"\s*:\s*"([^"]+)"', plan_entry['response'], re.DOTALL)
            pred_action_type = action_match.group(1).strip().strip(',').strip('"').lower() if action_match else ""
        if not len(pred_action_text) or pred_action_text == "none":
            value_match = re.search(r'"value"\s*:\s*"([^"]*)"', plan_entry['response'], re.DOTALL)
            pred_action_text = value_match.group(1).strip().strip(',').strip('"').lower() if value_match else ""
        # breakpoint()
        action_mapping = {
            'longpress': 'long_press',
            'openapp': 'open_app',
            'terminate': 'status',
            'write': 'type_text',
            'navigateback': 'navigate_back',
            'back': 'navigate_back',
            'home': 'navigate_home',
            'press_back': 'navigate_back',
            'complete': 'status',
            'click': 'click',
            'scroll': 'scroll',
            'wait': 'wait',
        }
        if pred_action_type in action_mapping:
            pred_action_type = action_mapping[pred_action_type]
        else:
            pass
            # print(f"Unknown predicted action type mapping: {pred_action_type} for gold action {gold_action}")
        
        if 'click' in pred_action_type:
            pred_action = {'action_type': 'click'}
        elif 'long_press' in pred_action_type:
            pred_action = {'action_type': 'long_press'}
        elif 'type_text' in pred_action_type:
            pred_action = {'action_type': 'type_text', 'text': pred_action_text}
        elif 'navigate_back' in pred_action_type:
            pred_action = {'action_type': 'navigate_back'}
        elif 'open_app' in pred_action_type:
            pred_action = {'action_type': 'open_app', 'app_name': pred_action_text}
        elif 'scroll' in pred_action_type:
            pred_action = {'action_type': 'scroll', 'direction': pred_action_text}
        elif 'swipe' in pred_action_type:
            direction = 'down'
            if ' up ' in pred_action_text:
                direction = 'down'
            elif ' down ' in pred_action_text:
                direction = 'up'
            elif ' left ' in pred_action_text:
                direction = 'right'
            elif ' right ' in pred_action_text:
                direction = 'left'
            pred_action = {'action_type': 'scroll', 'direction': direction}
        elif 'status' in pred_action_type:
            pred_action = {'action_type': 'status', 'goal_status': 'successful' if not 'fail' in pred_action_text else 'failed'}
        else:
            # print(f"Unknown predicted action type: {pred_action_type}")
            pred_action = {'action_type': pred_action_type}
        
        # print(plan_entry)
        # breakpoint()
        # print(pred_action, gold_action)
        GROUNDING_GT_TYPES = {'click', 'long_press', 'type_text'}
        gt_action_type = gold_action.get('action_type', '')
        if pred_action_type == gt_action_type:
            correct_types += 1
        gt_is_grounding = gt_action_type in GROUNDING_GT_TYPES
        if gt_is_grounding:
            grounding_steps += 1

        step_correct = False
        step_ground_correct = False
        pred_x, pred_y = None, None
        if gt_is_grounding and pred_action['action_type'] == gt_action_type:
            if 'UI-TARS' in plan_file:
                m = re.search(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", plan_entry['response'])
                if m:
                    x, y = int(m.group(1)), int(m.group(2))
                else:
                    x, y = 0, 0
            # elif 'GLM' in plan_file:
            #     x, y = plan_entry['point_2d']
            #     if 0 <= x <= 1000 and 0 <= y <= 1000:
            #         x = x / 1000 * width
            #         y = y / 1000 * height

            elif 'point_2d' in plan_entry and plan_entry['point_2d'] and plan_entry['point_2d'] != [-100, -100] and plan_entry['point_2d'] != [0, 0]:
                x, y = plan_entry['point_2d']
            else:
                output = extract_coord(plan_entry['response'])
                x, y = output[0]

            # if gt_action_type == 'type_text':
                # print(x, y, gold_action)
                # breakpoint()
            
            if args.relative_coord:
                x = x / 1000 * width
                y = y / 1000 * height

            scale = 1
            x /= scale
            y /= scale
            node, bbox = find_smallest_bbox_node(gold_action['x'], gold_action['y'], sample_entry['accessibility_tree'])
            # print(bounding_box_contains_point(bbox, x, y), x, y)
            if bbox and bounding_box_contains_point(bbox, x, y):
                if gold_action['action_type'] == 'type_text':
                    if gold_action['text'].lower() == pred_action_text:
                        step_correct = True
                        step_ground_correct = True
                else:
                    step_correct = True
                    step_ground_correct = True
        else:
            if all([gold_action[key].lower() == pred_action[key].lower() for key in gold_action if key in pred_action]):
                step_correct = True
            else:
                step_correct = False
        
        # check equivalent action
        if (pred_action_type == 'click' and gold_action['action_type'] == 'open_app') or \
        (pred_action_type == 'open_app' and gold_action['action_type'] == 'click'):
            # if pred_action_type == 'click':
  
            output = extract_coord(plan_entry['response'])
            x, y = output[0]
            if args.relative_coord:
                x = x / 1000 * width
                y = y / 1000 * height

            scale = 1
            pred_x = x / scale
            pred_y = y / scale

            element, _ = find_smallest_bbox_node(pred_x, pred_y, sample_entry['accessibility_tree'])
            if element:
                text = element.get('text', "")
                if text:
                    text = text.lower()
                content = element.get("content_description", "")
                if content:
                    content = content.lower()
                app_name = gold_action.get("app_name", "").lower()
                if (text and app_name in text) or (content and app_name in content):
                    step_correct = True

        if (pred_action_type == 'click' and gold_action['action_type'] == 'navigate_back') or \
        (pred_action_type == 'navigate_back' and gold_action['action_type'] == 'click'):
            output = extract_coord(plan_entry['response'])
            x, y = output[0]
            if args.relative_coord:
                x = x / 1000 * width
                y = y / 1000 * height

            scale = 1
            pred_x = x / scale
            pred_y = y / scale
            
            element, _ = find_smallest_bbox_node(pred_x, pred_y, sample_entry['accessibility_tree'])
            if element:
                text = element.get('text', "")
                if text:
                    text = text.lower()
                content = element.get("content_description", "")
                if content:
                    content = content.lower()
                if (text and "back" in text) or (content and "back" in content):
                    step_correct = True
        
        if step_correct:
            correct_steps += 1
        if step_ground_correct:
            correct_grounding_steps += 1
        # print(pred_action, pred_x, pred_y, gold_action, step_correct, step_ground_correct)

    results['correct_types'] = correct_types
    results["correct_steps"] = correct_steps
    results["total_steps"] = total_steps
    results["grounding_steps"] = grounding_steps
    results["correct_grounding_steps"] = correct_grounding_steps

    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and evaluate AndroidControl task steps")
    parser.add_argument("--sample_file", type=str, required=True, help="Path to the sample JSON file")
    parser.add_argument("--plan_file", type=str, required=True, help="Path to the plan JSONL file")
    parser.add_argument("--relative_coord", type=int, default=0, choices=[0, 1], help="Whether to use relative coordinates (0/1)")

    args = parser.parse_args()

    results = process_file(args.sample_file, args.plan_file)
    
    type_accuracy = results['correct_types'] / results['total_steps'] if results['total_steps'] > 0 else 0
    accuracy = results['correct_steps'] / results['total_steps'] if results['total_steps'] > 0 else 0
    grounding_accuracy = results['correct_grounding_steps'] / results['grounding_steps'] if results['grounding_steps'] > 0 else 0

    print(f"Type Acc: {type_accuracy:.2%} ({results['correct_types']}/{results['total_steps']})")
    print(f"Accuracy: {accuracy:.2%} ({results['correct_steps']}/{results['total_steps']})")
    print(f"Grounding Accuracy: {grounding_accuracy:.2%} ({results['correct_grounding_steps']}/{results['grounding_steps']})")
    # breakpoint()