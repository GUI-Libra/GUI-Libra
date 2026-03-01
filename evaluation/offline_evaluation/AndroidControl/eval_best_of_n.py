import json
from collections import defaultdict
import argparse
import re
import os

def bounding_box_contains_point(bbox, x, y):
    return bbox['x_min'] <= x <= bbox['x_max'] and bbox['y_min'] <= y <= bbox['y_max']

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
        

        # if 'GUI-R1' in plan_file and pred_action['action_type'] == 'scroll':
        #     # change reverse direction for GUI-R1
        #     if pred_action['direction'] == 'up':
        #         pred_action['direction'] = 'down'
        #     elif pred_action['direction'] == 'down':
        #         pred_action['direction'] = 'up'
        #     elif pred_action['direction'] == 'left':
        #         pred_action['direction'] = 'right'
        #     elif pred_action['direction'] == 'right':
        #         pred_action['direction'] = 'left'
        #     else:
        #         pass

        GROUNDING_GT_TYPES = {'click', 'long_press', 'type_text'}
        gt_action_type = gold_action.get('action_type', '')
        if gt_action_type  == pred_action_type:
            correct_types += 1
        gt_is_grounding = gt_action_type in GROUNDING_GT_TYPES
        if gt_is_grounding:
            grounding_steps += 1

        step_correct = False
        step_ground_correct = False

        if gt_is_grounding and pred_action['action_type'] == gt_action_type:
            if 'UI-TARS' in plan_file:
                m = re.search(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", plan_entry['response'])
                if m:
                    x, y = int(m.group(1)), int(m.group(2))
                else:
                    x, y = 0, 0
            elif 'point_2d' in plan_entry and plan_entry['point_2d']:
                x, y = plan_entry['point_2d']
            else:
                output = extract_coord(plan_entry['response'])
                x, y = output[0]

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
        # print(pred_action, gold_action, step_correct, step_ground_correct)

        results[(episode_id, step)] = {
            'correct_type': pred_action_type == gt_action_type,
            'correct_step': step_correct, 
            'correct_grounding_step': step_ground_correct, 
            'is_grounding_step': gt_is_grounding
            }

    results["correct_types"] = correct_types
    results["correct_steps"] = correct_steps
    results["total_steps"] = total_steps
    results["grounding_steps"] = grounding_steps
    results["correct_grounding_steps"] = correct_grounding_steps

    return results




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and evaluate AndroidControl task steps")
    parser.add_argument("--sample_file", type=str, required=True, help="Path to the sample JSON file")
    parser.add_argument("--plan_file", type=str, required=True, help="Path to the plan JSONL file")
    parser.add_argument("--file_num", type=int, default=1, help="Number of answer files to process (default: 1)")
    parser.add_argument("--relative_coord", action='store_true', help="Whether the predicted coordinates are relative (default: False)")
    args = parser.parse_args()

    results_list = []
    for i in range(0, args.file_num):
        plan_file = args.plan_file.replace('_0_', f'_{i}_')
        if not os.path.exists(plan_file):
            print(f"Plan file {plan_file} does not exist, skipping.")
            continue
        results = process_file(args.sample_file, plan_file)
        results_list.append(results)

    # calculate best of n results, correct if any of the n results are correct
    best_results = defaultdict(int)
    for results in results_list:
        for key, value in results.items():
            if isinstance(value, dict):
                if key not in best_results:
                    best_results[key] = {}
                for sub_key, sub_value in value.items(): 
                    best_results[key][sub_key] = best_results[key].get(sub_key, False) or sub_value


    # calculate the best-of-n accuracy
    correct_steps = sum(1 for v in best_results.values() if isinstance(v, dict) and v.get('correct_step', False))
    total_steps = len(best_results)
    correct_types = sum(1 for v in best_results.values() if isinstance(v, dict) and v.get('correct_type', False))
    grounding_steps = sum([1 for v in best_results.values() if isinstance(v, dict) and v.get('is_grounding_step', False)])
    accuracy = correct_steps / total_steps if total_steps > 0 else 0
    correct_grounding_steps = sum([float(v['correct_grounding_step']) for v in best_results.values() if isinstance(v, dict)])
    grounding_accuracy = correct_grounding_steps / grounding_steps if grounding_steps > 0 else 0
    print(f"Type Acc: {correct_types/total_steps:.2%} ({correct_types}/{total_steps})")
    print(f"Accuracy: {accuracy:.2%} ({correct_steps}/{total_steps})")
    print(f"Grounding Accuracy: {grounding_accuracy:.2%} ({correct_grounding_steps}/{grounding_steps})")