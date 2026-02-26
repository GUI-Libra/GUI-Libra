# Copyright 2024 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A Multimodal Autonomous Agent for Android (M3A)."""
import re
import os 
import io
import time
import json
import numpy as np
from PIL import Image
from openai import OpenAI

import base64

from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents import infer
from android_world.agents import m3a_utils
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils


# Plugin Functions
select_option_func = {
    "name": "browser.select_option",
    "description": "Select an option from a dropdown menu",
    "parameters": {
        "type": "object",
        "properties": {
            "x": {
                "type": "number",
                "description": "The x coordinate of the dropdown menu",
            },
            "y": {
                "type": "number",
                "description": "The y coordinate of the dropdown menu",
            },
            "value": {
                "type": "string",
                "description": "The value of the option to select",
            },
        },
        "required": ["x", "y", "value"],
    },
}

swipe_func = {
    "name": "mobile.swipe",
    "description": "Swipe on the screen",
    "parameters": {
        "type": "object",
        "properties": {
            "from_coord": {
                "type": "array",
                "items": {"type": "number"},
                "description": "The starting coordinates of the swipe",
            },
            "to_coord": {
                "type": "array",
                "items": {"type": "number"},
                "description": "The ending coordinates of the swipe",
            },
        },
        "required": ["from_coord", "to_coord"],
    },
}

home_func = {"name": "mobile.home", "description": "Press the home button"}

back_func = {"name": "mobile.back", "description": "Press the back button"}

wait_func = {
    "name": "mobile.wait",
    "description": "wait for the change to happen",
    "parameters": {
        "type": "object",
        "properties": {
            "seconds": {
                "type": "number",
                "description": "The seconds to wait",
            },
        },
        "required": ["seconds"],
    },
}

long_press_func = {
    "name": "mobile.long_press",
    "description": "Long press on the screen",
    "parameters": {
        "type": "object",
        "properties": {
            "x": {
                "type": "number",
                "description": "The x coordinate of the long press",
            },
            "y": {
                "type": "number",
                "description": "The y coordinate of the long press",
            },
        },
        "required": ["x", "y"],
    },
}

open_app_func = {
    "name": "mobile.open_app",
    "description": "Open an app on the device",
    "parameters": {
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string",
                "description": "The name of the app to open",
            },
        },
        "required": ["app_name"],
    },
}

agent_system_message = f"""You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.

You have access to the following functions:
- {json.dumps(swipe_func)}
- {json.dumps(home_func)}
- {json.dumps(back_func)}
- {json.dumps(wait_func)}
- {json.dumps(long_press_func)}
- {json.dumps(open_app_func)}
"""

user_instruction = """Please generate the next move according to the ui screenshot, instruction and previous actions.

Instruction: {}

Previous actions:
{}
"""

# Utils for Visual Grounding
PROMPT_PREFIX = '''
You are an agent who can operate an Android phone on behalf of a user. Based on user's goal/request, you may
- Answer back if the request/goal is a question (or a chat message), like user asks "What is my schedule for today?".
- Complete some tasks described in the requests/goals by performing actions (step by step) on the phone.

When given a user request, you will try to complete it step by step. At each step, you will be given the current screenshot and a history of what you have done (in text). Based on these pieces of information and the goal, you must choose to perform one of the actions in the following list (action description followed by the JSON format) by outputting the action in the JSON format.
- Today's date is 2023-10-15. Pay attention to time-related requirements in the instruction. 
- If you think the task has been completed, finish the task by using the status action with complete as goal_status: `{{"action_type": "status", "goal_status": "complete"}}`
- If you think the task is not feasible (including cases like you don't have enough information or cannot perform some necessary actions), finish by using the `status` action with infeasible as goal_status: `{{"action_type": "status", "goal_status": "infeasible"}}`
- Answer user's question: `{{"action_type": "answer", "text": "<answer_text>"}}`
- Click/tap on an element on the screen. Please write a description about the target element/position/area to help locate it: `{{"action_type": "click", "element": <description about the target element>}}`.
- Long press on an element on the screen, similar to the click action above: `{{"action_type": "long_press", "element": <description about the target element>}}`.
- Type text into a text field (this action contains clicking the text field, typing in the text, and pressing enter, so no need to click on the target field to start): `{{"action_type": "input_text", "text": <text_input>, "element": <description about the target element>}}`
- Press the Enter key: `{{"action_type": "keyboard_enter"}}`
- Navigate to the home screen: `{{"action_type": "navigate_home"}}`
- Navigate back: `{{"action_type": "navigate_back"}}`
- Scroll the screen or a scrollable UI element in one of the four directions, use the same element description as above if you want to scroll a specific UI element, leave it empty when scrolling the whole screen: `{{"action_type": "scroll", "direction": <up, down, left, right>, "element": <optional description about the target element>}}`
- Open an app (nothing will happen if the app is not installed. So always try this first if you want to open a certain app): `{{"action_type": "open_app", "app_name": <name>}}`
- Wait for the screen to update: `{{"action_type": "wait"}}`
'''

question_description = '''Please generate the next move according to the UI screenshot, instruction and previous actions.\n\nInstruction: {}\n\nInteraction History: {}\n'''


SUMMARY_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '''
The (overall) user goal/request is: {goal}
Now I want you to summarize the latest step.
You will be given the screenshot before you performed the action (which has a text label "before" on the bottom right), the action you chose (together with the reason) and the screenshot after the action was performed (A red dot is added to the screenshot if the action involves a target element/position/area, showing the located position. Carefully examine whether the red dot is pointing to the target element.).

This is the action you picked: {action}
Based on the reason: {reason}

By comparing the two screenshots and the action performed, give a brief summary of this step. This summary will be added to action history and used in future action selection, so try to include essential information you think that will be most useful for future action selections like what you intended to do, why, if it worked as expected, if not what might be the reason (be critical, the action/reason/locating might be wrong), what should/should not be done next, what should be the next step, and so on. Some more rules/tips you should follow:
- Keep it short (better less than 100 words) and in a single line
- Some actions (like `answer`, `wait`) don't involve screen change, you can just assume they work as expected.
- Given this summary will be added into action history, it can be used as memory to include information that needs to be remembered, or shared between different apps.
- If the located position is wrong, that is not your fault. You should try using another description style for this element next time.

Summary of this step: '''
)


def _generate_ui_element_description(
        ui_element: representation_utils.UIElement, index: int
) -> str:
    """Generate a description for a given UI element with important information.

    Args:
      ui_element: UI elements for the current screen.
      index: The numeric index for the UI element.

    Returns:
      The description for the UI element.
    """
    element_description = f'UI element {index}: {{"index": {index}, '
    if ui_element.text:
        element_description += f'"text": "{ui_element.text}", '
    if ui_element.content_description:
        element_description += (
            f'"content_description": "{ui_element.content_description}", '
        )
    if ui_element.hint_text:
        element_description += f'"hint_text": "{ui_element.hint_text}", '
    if ui_element.tooltip:
        element_description += f'"tooltip": "{ui_element.tooltip}", '
    element_description += (
        f'"is_clickable": {"True" if ui_element.is_clickable else "False"}, '
    )
    element_description += (
        '"is_long_clickable":'
        f' {"True" if ui_element.is_long_clickable else "False"}, '
    )
    element_description += (
        f'"is_editable": {"True" if ui_element.is_editable else "False"}, '
    )
    if ui_element.is_scrollable:
        element_description += '"is_scrollable": True, '
    if ui_element.is_focusable:
        element_description += '"is_focusable": True, '
    element_description += (
        f'"is_selected": {"True" if ui_element.is_selected else "False"}, '
    )
    element_description += (
        f'"is_checked": {"True" if ui_element.is_checked else "False"}, '
    )
    return element_description[:-2] + '}'


def _generate_ui_elements_description_list(
        ui_elements: list[representation_utils.UIElement],
        screen_width_height_px: tuple[int, int],
) -> str:
    """Generate concise information for a list of UIElement.

    Args:
      ui_elements: UI elements for the current screen.
      screen_width_height_px: The height and width of the screen in pixels.

    Returns:
      Concise information for each UIElement.
    """
    tree_info = ''
    for index, ui_element in enumerate(ui_elements):
        if m3a_utils.validate_ui_element(ui_element, screen_width_height_px):
            tree_info += _generate_ui_element_description(ui_element, index) + '\n'
    return tree_info


def _summarize_prompt(
        action: str,
        reason: str,
        goal: str,
        before_elements: str,
        after_elements: str,
) -> str:
    """Generate the prompt for the summarization step.

    Args:
      action: Action picked.
      reason: The reason to pick the action.
      goal: The overall goal.
      before_elements: Information for UI elements on the before screenshot.
      after_elements: Information for UI elements on the after screenshot.

    Returns:
      The text prompt for summarization that will be sent to gpt4v.
    """
    return SUMMARY_PROMPT_TEMPLATE.format(
        goal=goal,
        before_elements=before_elements,
        after_elements=after_elements,
        action=action,
        reason=reason,
    )


class SeeAct_V(base_agent.EnvironmentInteractingAgent):
    """M3A which stands for Multimodal Autonomous Agent for Android."""

    def __init__(
            self,
            env: interface.AsyncEnv,
            llm: infer.MultimodalLlmWrapper,
            name: str = 'M3A',
            wait_after_action_seconds: float = 2.0,
            grounding_model_address='http://localhost:8000/',
            grounding_model_api_key="token-abc123",
            grounding_model_name='osunlp/UGround-V1-7B',
            file_logger=None,
            save_img_dir: str = './save_img',
    ):
        """Initializes a M3A Agent.

        Args:
          env: The environment.
          llm: The multimodal LLM wrapper.
          name: The agent name.
          wait_after_action_seconds: Seconds to wait for the screen to stablize
            after executing an action
        """
        super().__init__(env, name)
        self.llm = llm
        self.history = []
        self.additional_guidelines = None
        self.wait_after_action_seconds = wait_after_action_seconds
        self.file_logger = file_logger
        self._save_img_dir = save_img_dir

        if grounding_model_name != 'self':
            self.grounding_model_client = OpenAI(
                base_url=f"{grounding_model_address}v1",
                api_key=grounding_model_api_key
            )
        else:
            self.grounding_model_client = None

        self.grounding_model_name = grounding_model_name

    def array_to_jpeg_bytes(image: np.ndarray) -> bytes:
        """Converts a numpy array into a byte string for a JPEG image."""
        image = Image.fromarray(image)
        in_mem_file = io.BytesIO()
        image.save(in_mem_file, format='JPEG')
        # Reset file pointer to start
        in_mem_file.seek(0)
        img_bytes = in_mem_file.read()
        return img_bytes

    def get_point_from_description(self, image: np.ndarray,description: str, ) -> tuple[int, int]:
        """Get the point from the description using the grounding model. This has been adapted to Qwen2-VL-based UGround. You may want to change the details of processing image and coordinates to fit your model.

        Args:
            description: The description of the point.
            image: The image to process.

        Returns:
            The (x, y) coordinates of the point.
        """

        def format_openai_template(description: str, base64_image):
            return [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                        {
                            "type": "text",
                            "text": f"""
          Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.

          - Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
          - If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
          - Your answer should be a single string (x, y) corresponding to the point of the interest.

          Description: {description}

          Answer:"""
                        },
                    ],
                },
            ]

        img = Image.fromarray(image)

        new_width = 882
        new_height = 1960
        width,height = img.size

        # print(width,height)

        img_resized = img.resize((new_width, new_height))

        if img_resized.mode == 'RGBA':
            img_resized = img_resized.convert('RGB')

        img_byte_arr = io.BytesIO()
        img_resized.save(img_byte_arr, format='JPEG')  # 调整质量以压缩图像
        image_bytes = img_byte_arr.getvalue()

        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        messages = format_openai_template(description, base64_image)

        completion =  self.grounding_model_client.chat.completions.create(
            model=self.grounding_model_name,
            messages=messages,
            temperature=0
        )


        response_text = completion.choices[0].message.content.strip()
        ratio_coords = eval(response_text)
        x_ratio, y_ratio = ratio_coords

        # 计算绝对坐标
        x_coord = round(x_ratio / 1000 * width)
        y_coord = round(y_ratio / 1000 * height)

        return (x_coord,y_coord)

    def set_task_guidelines(self, task_guidelines: list[str]) -> None:
        self.additional_guidelines = task_guidelines

    def reset(self, task_id, repeat_id, go_home_on_reset: bool = False):
        super().reset(go_home_on_reset)
        # Hide the coordinates on screen which might affect the vision model.
        # self.env.hide_automation_ui()
        self.task_idx = task_id
        self.history = []
        self.save_img_path = self._save_img_dir + f'/{task_id}_{repeat_id}'
        if not os.path.exists(self.save_img_path):
            os.makedirs(self.save_img_path)

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        step_data = {
            'raw_screenshot': None,
            'before_screenshot_with_som': None,
            'before_ui_elements': [],
            'after_screenshot_with_som': None,
            'action_prompt': None,
            'action_output': None,
            'action_output_json': None,
            'action_reason': None,
            'action_raw_response': None,
            'summary_prompt': None,
            'summary': None,
            'summary_raw_response': None,
        }
        if not self.file_logger:
            print('----------step ' + str(len(self.history) + 1))
        else:
            self.file_logger.info('----------step ' + str(len(self.history) + 1))

        # state = self.get_post_transition_state()
        # step_data['raw_screenshot'] = state.pixels.copy()
        step_data['raw_screenshot'] = self.env.get_screenshot()
        before_screenshot = step_data['raw_screenshot'].copy()
        step_data['before_screenshot_with_som'] = before_screenshot.copy()
        b64 = self.llm.encode_image(step_data['raw_screenshot'])


        history = "".join(
            f"\nStep {i+1}\nAction: {his['action']}\n" for i, his in enumerate(self.history)
        )

        query = user_instruction.format(goal, history)
        messages = [
                    {"role":"system",
                        "content":[{"type":"text",
                                    "text": agent_system_message}]},
                    {"role":"user", "content":[
                        {"type":"image_url",
                            "image_url":{"url":f"data:image/png;base64,{b64}",
                                        "detail":"high"}},
                        {"type":"text", "text": query}
                    ]}
                ]
        step_data['action_prompt'] = query
        action_output, is_safe, raw_response = self.llm.predict_mm_messages(
            messages
        )

        if is_safe == False:  # pylint: disable=singleton-comparison
            #  is_safe could be None
            action_output = f"""Reason: {m3a_utils.TRIGGER_SAFETY_CLASSIFIER}
Action: {{"action_type": "status", "goal_status": "infeasible"}}"""

        if not raw_response:
            raise RuntimeError('Error calling LLM in action selection phase.')
        step_data['action_output'] = action_output
        step_data['action_raw_response'] = raw_response.choices[0].message.content

        action = action_output.split('\nassistant')[0].lstrip('Action:').strip()
        try:
            action_type = action_output.split('(')[-2].split('.')[-1]
        except:
            if 'answer' in action_output.lower():
                action_type = 'answer'
            elif 'terminate' in action_output.lower():
                action_type = 'terminate'

        element_description = action

        value = action_output.split('(')[-1].split('=')[-1].split(')')[0].strip("'")

        # If the output is not in the right format, add it to step summary which
        # will be passed to next step and return.
        if not action:
            print('Action prompt output is not in the correct format.')
            step_data['summary'] = (
                'Output for action selection is not in the correct format, so no'
                ' action is performed.'
            )
            self.history.append(step_data)

            return base_agent.AgentInteractionResult(
                False,
                step_data,
            )
        if not self.file_logger:
            print('Action: ' + action_output)
        else:
            self.file_logger.info('Action: ' + action_output)

        step_data['action'] = action

        action_type_mapping = {
            "click": "click",
            'long_press': 'long_press',
            "write": "input_text",
            "scroll": "scroll",
            "back": "navigate_back",
            "home": "navigate_home",
            "wait": "wait",
            "terminate": "status",
            "answer": "answer",
            "open_app": "open_app",
            "swipe": "swipe",
        }
        if action_type.lower() in action_type_mapping:
            action_type = action_type_mapping[action_type.lower()]
        else:
            print(f"Unknown action type: {action_type}. Defaulting to 'unknown'.")
            action_type = "unknown"

        action_dict = {'action_type': action_type}
        if action_type == 'open_app':
            value = value.strip()
            action_dict['app_name'] = value
        elif action_type == 'input_text':
            action_dict['text'] = value.strip()
            action_dict['element'] = element_description.strip()
        elif action_type in ['long_press', 'click']:
            action_dict['element'] = element_description.strip()
        elif action_type == 'answer':
            action_dict['text'] = value.strip()
        elif action_type == 'status':
            action_dict['goal_status'] = 'complete'
        elif action_type == 'scroll':
            action_dict['element'] = element_description.strip()
            if 'right' in action_output.lower():
                action_dict['direction'] = 'right'
            else:
                action_dict['direction'] = 'down' 

        elif action_type == 'swipe':
            action_dict['element'] = element_description.strip()
            if 'left' in action_output.lower():
                action_dict['direction'] = 'left'
            else:
                action_dict['direction'] = 'up'

        if self.file_logger:
            self.file_logger.info('Action dict: ' + str(action_dict))
        else:
            print('Action dict: ' + str(action_dict))

        # save image
        from PIL import Image
        img = Image.fromarray(step_data['raw_screenshot'].astype(np.uint8))
        img.save(f'{self.save_img_path}/{len(self.history)+1}.png')

        import traceback
        try:
            converted_action = json_action.JSONAction(
                **action_dict,
            )
            step_data['action_output_json'] = converted_action

            if converted_action.element:
                match = re.search(r"x\s*=\s*([-\d.]+)\s*,\s*y\s*=\s*([-\d.]+)", action_output)
                if match:
                    width, height = img.size
                    x = float(match.group(1)) * width
                    y = float(match.group(2)) * height
                    converted_action.x, converted_action.y = x, y
                    if self.file_logger:
                        self.file_logger.info(
                            f'Action element coordinates: x={x}, y={y}'
                        )
                    else:
                        print(f'Action element coordinates: x={x}, y={y}')

                converted_action.element = None

        except Exception as e:  # pylint: disable=broad-exception-caught
            print('Failed to convert the output to a valid action.')
            print(traceback.print_exc())
            print(str(e))
            step_data['summary'] = (
                'Can not parse the output to a valid action. Please make sure to pick'
                ' the action from the list with required parameters (if any) in the'
                ' correct JSON format!'
            )
            self.history.append(step_data)

            return base_agent.AgentInteractionResult(
                False,
                step_data,
            )

        if converted_action.action_type == 'status':
            if converted_action.goal_status == 'infeasible':
                print('Agent stopped since it thinks mission impossible.')
            step_data['summary'] = 'Agent thinks the request has been completed.'
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(
                True,
                step_data,
            )

        if converted_action.action_type == 'answer':
            print('Agent answered with: ' + converted_action.text)

        try:
            self.env.execute_action(converted_action)
        except Exception as e:  # pylint: disable=broad-exception-caught
            print('Failed to execute action.')
            print(str(e))
            step_data['summary'] = (
                'Can not execute the action, make sure to select the action with'
                ' the required parameters (if any) in the correct JSON format!'
            )
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(
                False,
                step_data,
            )

        time.sleep(self.wait_after_action_seconds)

        # state = self.env.get_state(wait_to_stabilize=False)

        # after_screenshot = state.pixels.copy()
        after_screenshot = self.env.get_screenshot()
        if converted_action.x:
            m3a_utils.add_ui_element_dot(
                before_screenshot,
                target_element=[round(converted_action.x), round(converted_action.y)] if converted_action.x else None

            )

        step_data['before_screenshot_with_som'] = before_screenshot.copy()
        m3a_utils.add_screenshot_label(after_screenshot, 'after')
        step_data['after_screenshot_with_som'] = after_screenshot.copy()


        self.history.append(step_data)
        return base_agent.AgentInteractionResult(
            False,
            step_data,
        )
