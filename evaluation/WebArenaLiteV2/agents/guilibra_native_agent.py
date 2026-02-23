import json
import logging
import re
from typing import Dict, List, Tuple
from agents.ui_agent import UIAgent
from agents.native_agent import OpenCUANativeAgent
from utils.misc import call_llm_safe, smart_resize, IMAGE_FACTOR
import ast

logger = logging.getLogger()

class GUILibraNativeAgent(OpenCUANativeAgent):
    def __init__(
        self,
        engine_params: Dict,
        platform: str = "web",
        width: int = 1600,
        height: int = 2560,
    ):
        super().__init__(engine_params=engine_params, platform=platform, width=width, height=height)


    def parse_plan(self, plan: str) -> tuple[str, str, str]:
        """
        Parse the plan string to extract thinking, operation and action parts.

        Args:
            plan: A string containing structured sections.

        Returns:
            A tuple of (thought, operation, action) strings.
        """
        # Extract thinking section
        if self.model_backbone == "qwen3vl":
            think_pattern = r"<thinking>(.*?)</thinking>"
        else:
            think_pattern = r"<think>(.*?)</think>"

        think_match = re.search(think_pattern, plan, re.DOTALL)
        thought = think_match.group(1).strip() if think_match else ""

        # Extract answer section
        answer_match = re.search(r"<answer>(.*?)</answer>", plan, re.DOTALL)
        action = answer_match.group(1).strip() if answer_match else ""

        # Try to extract description from JSON for operation
        # We look for "action_description" or "action_type" as fallback
        operation = ""
        if action:
            op_match = re.search(r'"action_description"\s*:\s*"([^"]+)"', action)
            if op_match:
                operation = op_match.group(1)
            else:
                # Fallback to action_type if description is missing
                type_match = re.search(r'"action_type"\s*:\s*"([^"]+)"', action)
                if type_match:
                    operation = type_match.group(1)
            
            return thought, operation, action

    def generate_next_action(self, instruction: str, obs: Dict) -> Tuple[Dict, List]:
        """
        Generate the next action based on the current observation and instruction.

        Args:
            instruction: User's task instruction
            obs: Current observation from the environment

        Returns:
            Tuple containing execution info and action list
        """
        # Encode screenshot
        image_content = obs["screenshot"]
        base64_image, img_size = self.native_agent.encode_image(image_content)

        # Build history of previous operations (limited to last 15)
        previous_operations = ""
        for i, operation in enumerate(self.previous_operations_list[-15:]):
            previous_operations += f"Step {i + 1}\n Action: {operation}\n"
            
        previous_operations = "None" if previous_operations == "" else previous_operations

        # Format user prompt
        img_size_string = '(original image size {}x{})'.format(img_size[0], img_size[1])
        user_prompt = self.user_instruction_template.format(
            screensize=img_size_string, instruction=instruction, actions=previous_operations
        )
        if self.model_backbone == "qwen3vl":
            think_tag = ["<thinking>", "</thinking>"]
        else:
            think_tag = ["<think>", "</think>"]
        user_prompt += f"""\n\nThe response should be structured in the following format, make sure the output between <answer> and </answer> is a valid JSON object. Regarding the key "point_2d", please provide the coordinates on the screen where the action is to be performed; if not applicable, use [-100, -100]:
{think_tag[0]}Your step-by-step thought process here...{think_tag[1]}
<answer>
{{
  "action_description": "the description of the action to perform, summarized in one sentence",
  "action_type": "the type of action to perform. Please follow the system prompt for available actions.",
  "value": "the input text or direction ('up', 'down', 'left', 'right') for the 'scroll' action, if applicable; otherwise, use 'None'",
  "point_2d": [x, y]
}}
</answer>
"""

        self.messages[1]["content"][1]["text"] = user_prompt

        self.messages[1]["content"][0]["image_url"][
            "url"
        ] = f"data:image/png;base64,{base64_image}"

        # Send message to language model
        self.native_agent.replace_messages(self.messages)
        logger.info("Input Messages: %s", self.messages)
        plan = call_llm_safe(self.native_agent)
        logger.info("API Response Plan: %s", plan)

        # Parse response according to expected format
        thought, operation, actions = self.parse_plan(plan)
        
        try:
            # Parse JSON action
            # Clean up json string if needed (sometimes models add markdown code blocks)
            json_str = actions
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            data = json.loads(json_str)
            
            # Update operation if we have a better description now
            if "action_description" in data:
                operation = data["action_description"]
            
            action_type = data.get("action_type")
            value = data.get("value")
            point_2d = data.get("point_2d")
            
            # Construct function call string for existing parser logic
            if action_type in ["Click", "Select"]:
                if point_2d and isinstance(point_2d, list) and len(point_2d) == 2:
                    # Assuming point_2d is [x, y] in pixels
                    x, y = point_2d
                    func_call = f"click(x={x}, y={y})"
                else:
                    raise ValueError("Invalid point_2d: {}".format(point_2d))
                exec_actions = [self.parse_exec_action(func_call)]
                    
            elif action_type == "Write":
                exec_actions = []
                # point_2d is a valid target position
                if point_2d and isinstance(point_2d, list) and len(point_2d) == 2 and (point_2d[0] != -100 and point_2d[1] != -100):
                    # point_2d is [x, y] in pixels
                    x, y = point_2d
                    func_call = f"click(x={x}, y={y})"
                    exec_actions.append(self.parse_exec_action(func_call))

                # value is the text
                safe_value = value.replace('"', '\\"').replace("'", "\\'")
                func_call = f"write(message=\"{safe_value}\")"
                exec_actions.append(self.parse_exec_action(func_call))

                # append an enter key press
                # func_call = "press(keys=\"enter\")"
                # exec_actions.append(self.parse_exec_action(func_call))
                
            elif action_type == "KeyboardPress":
                safe_value = value.replace('"', '\\"').replace("'", "\\'").lower()
                func_call = f"press(keys=\"{safe_value}\")"
                exec_actions = [self.parse_exec_action(func_call)]
                
            elif action_type == "Scroll":
                direction = value.lower() if value else "down"
                # Map scroll direction to swipe: up->down, down->up, left->right, right->left
                mapping = {"up": "down", "down": "up", "left": "right", "right": "left"}
                target_direction = mapping.get(direction, "down") # default to swipe up (scroll down) if unknown? Or stick to mapped default
                func_call = f"swipe(direction=\"{target_direction}\")"
                exec_actions = [self.parse_exec_action(func_call)]
                
            elif action_type == "Back" or (action_type == "Navigate" and value == "Back"):
                func_call = "back()"
                exec_actions = [self.parse_exec_action(func_call)]
            
            elif action_type == "Answer":
                safe_value = value.replace('"', '\\"').replace("'", "\\'")
                if not safe_value:
                    safe_value = "Task Completed"
                func_call = f"answer(\"{safe_value}\")"
                exec_actions = [self.parse_exec_action(func_call)]

            elif action_type == "terminate":
                safe_value = value.replace('"', '\\"').replace("'", "\\'")
                if not safe_value:
                    safe_value = "Task Completed"
                func_call = f"terminate(status=\"success\", info=\"{safe_value}\")"
                exec_actions = [self.parse_exec_action(func_call)]
            else:
                logger.warning(f"Error in parsing guipivot JSON: {data}")
                func_call = "wait(seconds=1)"
                exec_actions = [self.parse_exec_action(func_call)]

        except Exception as e:
            logger.error(f"Error parsing guipivot JSON: {e}")
            exec_actions = [{"name": "wait", "parameters": {"seconds": 1}}]

        self.previous_operations_list.append(operation)

        # Prepare execution info
        exec_info = {"thought": thought, "operation": operation, "actions": actions}

        return exec_info, exec_actions
