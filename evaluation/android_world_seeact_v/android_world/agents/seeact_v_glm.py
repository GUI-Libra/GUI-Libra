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
import os
import re 
import io
import abc 
import time
import json
from typing import Any
import dataclasses
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


def get_mobile_prompt(task, history):
    app_names = [
        "Google Chrome",
        "Google Chat",
        "Settings",
        "YouTube",
        "Google Play",
        "Gmail",
        "Google Maps",
        "Google Photos",
        "Google Calendar",
        "Camera",
        "Audio Recorder",
        "Google Drive",
        "Google Keep",
        "Grubhub",
        "Tripadvisor",
        "Starbucks",
        "Google Docs",
        "Google Sheets",
        "Google Slides",
        "Clock",
        "Google Search",
        "Contacts",
        "Facebook",
        "WhatsApp",
        "Instagram",
        "Twitter",
        "Snapchat",
        "Telegram",
        "LinkedIn",
        "Spotify",
        "Netflix",
        "Amazon Shopping",
        "TikTok",
        "Discord",
        "Reddit",
        "Pinterest",
        "Android World",
        "Files",
        "Markor",
        "Clipper",
        "Messages",
        "Simple SMS Messenger",
        "Dialer",
        "Simple Calendar Pro",
        "Simple Gallery Pro",
        "Miniwob",
        "Simple Draw Pro",
        "Pro Expense",
        "Broccoli",
        "CAA",
        "OsmAnd",
        "Tasks",
        "Open Tracks Sports Tracker",
        "Joplin",
        "VLC",
        "Retro Music",
    ]

    prompt = f"""You are an agent who can operate an Android phone on behalf of a user. Based on user's goal/request, you may
- Answer back if the request/goal is a question (or a chat message), like user asks "What is my schedule for today?".
- Complete some tasks described in the requests/goals by performing actions (step by step) on the phone.

When given a user request, you will try to complete it step by step. At each step, you will be given the current screenshot (including the original screenshot and the same screenshot with bounding boxes and numeric indexes added to some UI elements) and a history of what you have done (in text). Based on these pieces of information and the goal, you must choose to perform one of the action in the following list (action description followed by the JSON format) by outputting the action in the correct JSON format.
- Today's date is 2023-10-15. Pay attention to time-related requirements in the instruction. 
- If you think the task has been completed, finish the task by using the status action with complete as goal_status: `{{"action_type": "status", "goal_status": "complete"}}`
- If you think the task is not feasible (including cases like you don't have enough information or can not perform some necessary actions), finish by using the `status` action with infeasible as goal_status: `{{"action_type": "status", "goal_status": "infeasible"}}`
- Answer user's question: `{{"action_type": "answer", "text": "<answer_text>"}}`
-- You should only answer once in one command. If you needs multiple pieces of information to answer the question, you should gather the information in "Memory" and answer the question when you have enough information.
- Click/tap on an element on the screen. Use the box_2d to indicate which element you want to click: `{{"action_type": "click", "box_2d": [[,,,]]}}`. The box_2d should be [[xmin,ymin,xmax,ymax]] normalized to 0-999, indicating the position of the element.
- Long press on an element on the screen, similar with the click action above, use the box_2d to indicate which element you want to long press: `{{"action_type": "long_press", "box_2d": [[,,,]]}}`.
- Type text into a text field (this action contains clicking the text field, typing in the text and pressing the enter, so no need to click on the target field to start), use the box_2d to indicate the target text field. The text to be input can be from the command, the memory, or the current screen: `{{"action_type": "input_text", "text": <text_input>, "box_2d": [[,,,]], 'override': True/False}}`. If override is True, the text field will be cleared before typing.
- Press the Enter key: `{{"action_type": "keyboard_enter"}}`
- Navigate to the home screen: `{{"action_type": "navigate_home"}}`
- Navigate back: `{{"action_type": "navigate_back"}}`
- Swipe the screen or a scrollable UI element in one of the four directions, use the box_2d as above if you want to swipe a specific UI element, leave it empty when swipe the whole screen: `{{"action_type": "swipe", "direction": <up, down, left, right>, "box_2d": [[,,,]](optional)}}`. 
- Open an app (nothing will happen if the app is not installed): `{{"action_type": "open_app", "app_name": <name>}}`
-- supported app_names: {",".join(app_names)}
- Wait for the screen to update: `{{"action_type": "wait"}}`

The current user goal/request is: {task}

Here is a history of what you have done so far:
"""

    history_str = ""
    if len(history) == 0:
        history_str = "You just started, no action has been performed yet."
    else:
        for idx, h in enumerate(history):
            history_str += f"Step {idx+1}:\nThough: {h['reason']}\nAction: {h['action']}\n\n"

    prompt += history_str + "\n"

    prompt += """The current screenshot is given to you. 
Here are some useful guidelines you need to follow:
General:
- Usually there will be multiple ways to complete a task, pick the easiest one. Also when something does not work as expected (due to various reasons), sometimes a simple retry can solve the problem, but if it doesn't (you can see that from the history), SWITCH to other solutions.
- Sometimes you may need to navigate the phone to gather information needed to complete the task, for example if user asks "what is my schedule tomorrow", then you may want to open the calendar app (using the `open_app` action), look up information there, answer user's question (using the `answer` action) and finish (using the `status` action with complete as goal_status).
- For requests that are questions (or chat messages), remember to use the `answer` action to reply to user explicitly before finish! Merely displaying the answer on the screen is NOT sufficient (unless the goal is something like "show me ...").
- If the desired state is already achieved (e.g., enabling Wi-Fi when it's already on), you can just complete the task.
- If we say that two items are duplicated, in most cases we require that all of their attributes are exactly the same, not just the name.
Text Related Operations:
- Normally to select certain text on the screen: <i> Enter text selection mode by long pressing the area where the text is, then some of the words near the long press point will be selected (highlighted with two pointers indicating the range) and usually a text selection bar will also appear with options like `copy`, `paste`, `select all`, etc. <ii> Select the exact text you need. Usually the text selected from the previous step is NOT the one you want, you need to adjust the range by dragging the two pointers. If you want to select all text in the text field, simply click the `select all` button in the bar.
- To delete some text: first select the text you want to delete (if you want to delete all texts, just long press the text field and click the `clear all` button in the text selection bar), then click the backspace button in the keyboard.
- To copy some text: first select the exact text you want to copy, which usually also brings up the text selection bar, then click the `copy` button in bar.
- To paste text into a text box, first long press the text box, then usually the text selection bar will appear with a `paste` button in it.
- When typing into a text field, sometimes an auto-complete dropdown list will appear. This usually indicating this is a enum field and you should try to select the best match by clicking the corresponding one in the list.
Action Related:
- Use the `input_text` action whenever you want to type something (including password) instead of clicking characters on the keyboard one by one. Sometimes there is some default text in the text field you want to type in, remember to delete them before typing.
- Consider exploring the screen by using the `swipe` action with different directions to reveal additional content.
- The direction parameter for the `swipe` action can be confusing sometimes as it's opposite to swipe, for example, to view content at the bottom, the `swipe` direction should be set to "up". It has been observed that you have difficulties in choosing the correct direction, so if one does not work, try the opposite as well.
- To open an app if you can not find its icon, you can first press home (if necessary) and swipe up to the app drawer.
- Swipe up means swiping from bottom to top, swipe down means swiping from top to bottom, swipe left means swiping from right to left, swipe right means swiping from left to right.
- Use the `navigate_back` action to close/hide the soft keyboard.

Now output: 
1. Memory: important information you want to remember for the future actions. The memory should be only contents on the screen that will be used in the future actions. It should satisfy that: you cannot determine one or more future actions without this memory. 
2. Reason: the reason for the action and the memory. Your reason should include, but not limited to:- the content of the GUI, especially elements that are tightly related to the user goal- the step-by-step thinking process of how you come up with the new action. 
3. Action: the action you want to take, in the correct JSON format. The action should be one of the above list.

Your answer should look like:
Memory: ...
Reason: ...
Action: {"action_type":...}"""

    return prompt

def parse_mobile_response(response):
    if '<answer>' in response:
        # extract the answer from the response
        answer_start = response.index('<answer>') + len('<answer>')
        answer = response[answer_start:].strip()

    pattern = r"Memory:(.*?)Reason:(.*?)Action:(.*)"
    match = re.search(pattern, answer, re.DOTALL)
    if not match:
        return None

    memory = match.group(1).strip()
    reason = match.group(2).strip()
    action = match.group(3).strip()

    if "<|begin_of_box|>" in action:
        action = action[
            action.index("<|begin_of_box|>") + len("<|begin_of_box|>") : action.rindex(
                "<|end_of_box|>"
            )
        ]

    parsed_action = None
    if action.startswith("{"):
        parsed_action = json.loads(action)

    return {
        "memory": memory,
        "reason": reason,
        "action": action,
        "parsed_action": parsed_action,
    }

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


@dataclasses.dataclass()
class AgentInteractionResult:
  """Result of a single agent interaction with the environment.

  Attributes:
    done: Whether the agent indicates the entire session is done; i.e. this is
      the last interaction with the environment and the session will terminate.
    data: Environment and agent data from interaction.
  """

  done: bool
  data: dict[str, Any]

class EnvironmentInteractingAgent(abc.ABC):
  """Base class for an agent that directly interacts with and acts on the environment.

  This class provides flexibility in agent design, allowing developers to define
  custom action spaces and interaction methods without being confined to a
  specific approach.
  """

  def __init__(
      self,
      env: interface.AsyncEnv,
      name: str = '',
      transition_pause: float | None = 1.0,
  ):
    """Initializes the agent.

    Args:
      env: The environment.
      name: The agent name.
      transition_pause: The pause before grabbing the state. This is required
        because typically the agent is grabbing state immediatley after an
        action and the screen is still changing. If `None` is provided, then it
        uses "auto" mode which dynamically adjusts the wait time based on
        environmental feedback.

    Raises:
      ValueError: If the transition pause is negative.
    """
    self._env = env
    self._name = name
    if transition_pause is not None and transition_pause < 0:
      raise ValueError(
          f'transition_pause must be non-negative, got {transition_pause}'
      )
    self._transition_pause = transition_pause

    self._max_steps = None

  @property
  def transition_pause(self) -> float | None:
    return self._transition_pause

  @transition_pause.setter
  def transition_pause(self, transition_pause: float | None) -> None:
    self._transition_pause = transition_pause

  @property
  def env(self) -> interface.AsyncEnv:
    return self._env

  @env.setter
  def env(self, env: interface.AsyncEnv) -> None:
    self._env = env

  def set_max_steps(self, max_steps: int) -> None:
    self._max_steps = max_steps

  def reset(self, go_home: bool = False) -> None:
    """Resets the agent."""
    self.env.reset(go_home=go_home)

  @abc.abstractmethod
  def step(self, goal: str) -> AgentInteractionResult:
    """Performs a step of the agent on the environment.

    Args:
      goal: The goal.

    Returns:
      Done and agent & observation data.
    """

  @property
  def name(self) -> str:
    return self._name

  @name.setter
  def name(self, name: str) -> None:
    self._name = name

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
            use_its_own_grounding_model: bool = False
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
        self.use_its_own_grounding_model = use_its_own_grounding_model

        self.grounding_model_client = OpenAI(
            base_url=f"{grounding_model_address}v1",
            api_key=grounding_model_api_key
        )




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

        img = Image.fromarray(image.astype(np.uint8))

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


        before_screenshot = self.env.get_screenshot()
        step_data['raw_screenshot'] = before_screenshot.copy()
        step_data['before_screenshot_with_som'] = before_screenshot.copy()

        action_prompt = get_mobile_prompt(goal, self.history)
        step_data['action_prompt'] = action_prompt
        action_output, is_safe, raw_response = self.llm.predict_mm(
            action_prompt,
            [
                step_data['raw_screenshot'],
                # before_screenshot,
            ],
        )

        response_dict = parse_mobile_response(action_output)
        memory = response_dict['memory'] 
        reason = response_dict['reason']
        action = response_dict['action']
        converted_action = response_dict['parsed_action']

        if not raw_response:
            raise RuntimeError('Error calling LLM in action selection phase.')
        step_data['action_output'] = action_output
        step_data['action_raw_response'] = raw_response

        # If the output is not in the right format, add it to step summary which
        # will be passed to next step and return.
        if (not reason) or (not action):
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
            print('Reason: ' + reason)
            print('Action: ' + action)
        else:
            self.file_logger.info('Reason: ' + reason)
            self.file_logger.info('Action: ' + action)

        # save image
        img = Image.fromarray(step_data['raw_screenshot'].astype(np.uint8))
        # save image to the path of the current process
        img.save(f'{self.save_img_path}/{len(self.history)+1}.png')
        
        step_data['action_reason'] = reason
        step_data['action'] = action.strip()
        import traceback
        try:
            if 'box_2d' in converted_action:
                # Convert box_2d to x, y coordinates
                box_2d = converted_action['box_2d']
                if len(box_2d) == 1:
                    box_2d = box_2d[0]

                if len(box_2d) == 2:
                    converted_action['x'] = box_2d[0]
                    converted_action['y'] = box_2d[1]
                elif len(box_2d) == 4:
                    converted_action['x'] = (box_2d[0] + box_2d[2]) / 2
                    converted_action['y'] = (box_2d[1] + box_2d[3]) / 2
                
                if 'x' and 'y' in converted_action:
                    converted_action['x'] = int(converted_action['x'] / 1000 * img.size[0])
                    converted_action['y'] = int(converted_action['y'] / 1000 * img.size[1])
                breakpoint()
                converted_action.pop('box_2d', None)
                

            converted_action = json_action.JSONAction(
                **converted_action
            )
            step_data['action_output_json'] = converted_action

            if converted_action.element and converted_action.element != 'None':
                converted_action.x, converted_action.y = self.get_point_from_description(step_data['raw_screenshot'],
                                                                 converted_action.element)
                converted_action.element = None
            else:
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
            # print('Executing action: ' + str(converted_action))
            self.env.execute_action(converted_action)
            if converted_action.action_type == 'answer':
                self.history.append(step_data)
                return base_agent.AgentInteractionResult(
                    True,
                    step_data,
                )
        except Exception as e:  # pylint: disable=broad-exception-caught
            if not self.file_logger:
                print('Failed to execute action.')
                print(str(e))
            else:
                self.file_logger.error('Failed to execute action.')
                self.file_logger.error(str(e))

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
