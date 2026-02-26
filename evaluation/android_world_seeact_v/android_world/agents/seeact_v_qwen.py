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
import io
import os
import abc 
import time
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

# Utils for Visual Grounding
PROMPT_PREFIX = '''
You are an agent who can operate an Android phone on behalf of a user. Based on user's goal/request, you may
- Answer back if the request/goal is a question (or a chat message), like user asks "What is my schedule for today?".
- Complete some tasks described in the requests/goals by performing actions (step by step) on the phone.

When given a user request, you will try to complete it step by step. At each step, you will be given the current screenshot and a history of what you have done (in text). Based on these pieces of information and the goal, you must choose to perform one of the actions in the following list (action description followed by the JSON format) by outputting the action in the JSON format.
- If you think the task has been completed, finish the task by using the status action with complete as goal_status: `{{"action_type": "status", "goal_status": "complete"}}`
- If you think the task is not feasible (including cases like you don't have enough information or cannot perform some necessary actions), finish by using the `status` action with infeasible as goal_status: `{{"action_type": "status", "goal_status": "infeasible"}}`
- Answer user's question: `{{"action_type": "answer", "text": "<answer_text>"}}`
- Click/tap on an element on the screen. Please write a description about the target element/position/area to help locate it: `{{"action_type": "click", "element": <description about the target element>, "coordinates": <coordinates>}}`.
- Long press on an element on the screen, similar to the click action above: `{{"action_type": "long_press", "element": <description about the target element>, "coordinates": <coordinates>}}`.
- Type text into a text field (this action contains clicking the text field, typing in the text, and pressing enter, so no need to click on the target field to start): `{{"action_type": "input_text", "text": <text_input>, "element": <description about the target element>}}`
- Press the Enter key: `{{"action_type": "keyboard_enter"}}`
- Navigate to the home screen: `{{"action_type": "navigate_home"}}`
- Navigate back: `{{"action_type": "navigate_back"}}`
- Scroll the screen or a scrollable UI element in one of the four directions, use the same element description as above if you want to scroll a specific UI element, leave it empty when scrolling the whole screen: `{{"action_type": "scroll", "direction": <up, down, left, right>, "element": <optional description about the target element>, "coordinates": <optional coordinates>}}`
- Open an app (nothing will happen if the app is not installed. So always try this first if you want to open a certain app): `{{"action_type": "open_app", "app_name": <name>}}`
- Wait for the screen to update: `{{"action_type": "wait"}}`
'''


GUIDANCE = (
    '''Here are some useful guidelines you must follow:
General:
- Today's date is 2023-10-15. Pay attention to time-related requirements in the instruction. 
- Make sure you understand the task goal to avoid wrong actions.
- Make sure you carefully examine the the current screenshot. Sometimes the summarized history might not be reliable, over-claiming some effects.
- Pay attention to the screenshot. Make sure you issue a valid action given the current observation, especially for actions involving a specific element. The element you describe must be something actually in the screenshot right now, and make sure your description is sufficient for humans to locate it from the screenshot. Also, do not generate a same description consecutively for an target element. Always try to use different descriptions to help humans locate it from the screen.
- Usually there will be multiple ways to complete a task, pick the easiest one. Also when something does not work as expected (due to various reasons), sometimes a simple retry can solve the problem, but if it doesn't (you can see that from the history), SWITCH to other solutions. If you fall into obvious failure loops, please stop the action sequences and try another way to complete your intention.
- Sometimes you may need to navigate the phone to gather information needed to complete the task, for example if user asks "what is my schedule tomorrow", then you may want to open the calendar app (using the `open_app` action), look up information there, answer user's question (using the `answer` action) and finish (using the `status` action with complete as goal_status).
- For requests that are questions (or chat messages), remember to use the `answer` action to reply to user explicitly before finish! Merely displaying the answer on the screen is NOT sufficient (unless the goal is something like "show me ..."). REMEMBER to indicate "complete" status after you correctly answering the question if the goal is finished.
- If the desired state is already achieved (e.g., enabling Wi-Fi when it's already on), you can just complete the task.

Action Related:
- ALWAYS Use the `open_app` action whenever you want to open an app (nothing will happen if the app is not installed)! Otherwise you may open a wrong app asked by the task! please do not use the app drawer to open an app unless all other ways have failed. The correct way to open app drawer is to SCROLL DOWN (NOT UP) on the home screen (Use this only if the 'open_app' operation fails).
- Use the `input_text` action whenever you want to type something (including password) instead of clicking characters on the keyboard one by one. Sometimes there is some default text in the text field you want to type in, remember to delete them before typing.
- For `click`, `long_press` and `input_text`, make sure your target element/area/position is visible in the current screenshot, and make sure your description is sufficient enough for human to locate it.
- Consider exploring the screen by using the `scroll` action with different directions to reveal additional content.
- The direction parameter for the `scroll` action can be confusing sometimes as it's opposite to swipe, for example, to view content at the bottom, the `scroll` direction should be set to "down". It has been observed that you have difficulties in choosing the correct direction, so if one does not work, try the opposite as well.

Text Related Operations:
- Normally to select certain text on the screen: <i> Enter text selection mode by long pressing the area where the text is, then some of the words near the long press point will be selected (highlighted with two pointers indicating the range) and usually a text selection bar will also appear with options like `copy`, `paste`, `select all`, etc. <ii> Select the exact text you need. Usually the text selected from the previous step is NOT the one you want, you need to adjust the range by dragging the two pointers. If you want to select all text in the text field, simply click the `select all` button in the bar.
- At this point, you don't have the ability to drag something around the screen, so in general you can not select arbitrary text.
- To delete some text: the most traditional way is to place the cursor at the right place and use the backspace button in the keyboard to delete the characters one by one (can long press the backspace to accelerate if there are many to delete). Another approach is to first select the text you want to delete, then click the backspace button in the keyboard.
- To copy some text: first select the exact text you want to copy, which usually also brings up the text selection bar, then click the `copy` button in bar.
- To paste text into a text box, first long press the text box, then usually the text selection bar will appear with a `paste` button in it.
- When typing into a text field, sometimes an auto-complete dropdown list will appear. This usually indicating this is a enum field and you should try to select the best match by clicking the corresponding one in the list.'''
)


ACTION_SELECTION_PROMPT_TEMPLATE_LOCATE = (
    PROMPT_PREFIX
    + '''
The current user goal/request is: {goal}

Here is a history of what you have done so far:
{history}

The current screenshot is also given to you.
'''
    + GUIDANCE
    + '''
Now output an action from the above list in the correct JSON format, following the reason why you do that. Your answer should look like:
Reason: ...
Action: {{"action_type":...}}

Your Answer:
'''
)


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

Output a short Summary of this step instead of a json action: '''
)

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


def _action_selection_prompt_locate(
    goal: str,
    history: list[str],
    ui_elements: str,
    additional_guidelines: list[str] | None = None,
) -> str:
  """Generate the prompt for the action selection.

  Args:
    goal: The current goal.
    history: Summaries for previous steps.
    ui_elements: A list of descriptions for the UI elements.
    additional_guidelines: Task specific guidelines.

  Returns:
    The text prompt for action selection that will be sent to gpt4v.
  """
  if history:
    history = '\n'.join(history)
  else:
    history = 'You just started, no action has been performed yet.'

  extra_guidelines = ''
  if additional_guidelines:
    extra_guidelines = 'For The Current Task:\n'
    for guideline in additional_guidelines:
      extra_guidelines += f'- {guideline}\n'

  return ACTION_SELECTION_PROMPT_TEMPLATE_LOCATE.format(
      goal=goal,
      history=history,
      additional_guidelines=extra_guidelines,
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


class SeeAct_V(EnvironmentInteractingAgent):
    """M3A which stands for Multimodal Autonomous Agent for Android."""

    def __init__(
            self,
            env,
            llm: infer.MultimodalLlmWrapper,
            name: str = 'M3A',
            wait_after_action_seconds: float = 2.0,
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
        self.additional_guidelines = []
        self.wait_after_action_seconds = wait_after_action_seconds
        self.file_logger = file_logger
        self._save_img_dir = save_img_dir

    def array_to_jpeg_bytes(image: np.ndarray) -> bytes:
        """Converts a numpy array into a byte string for a JPEG image."""
        image = Image.fromarray(image)
        in_mem_file = io.BytesIO()
        image.save(in_mem_file, format='JPEG')
        # Reset file pointer to start
        in_mem_file.seek(0)
        img_bytes = in_mem_file.read()
        return img_bytes


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
        b64 = self.llm.encode_image(step_data['raw_screenshot'])

 
        history = [
            f"\nStep {i+1} Action: {his['action']}. {'Summary: ' + his['summary'] if ('summary' in his and his['summary'] != None) else ''}\n" for i, his in enumerate(self.history)
        ]

        query = _action_selection_prompt_locate(
            goal,
            history,
            None,
            self.additional_guidelines,
        )
        # print(query)
        messages = [
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
        step_data['action_raw_response'] = raw_response

        reason, action = m3a_utils.parse_reason_action_output(action_output)

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
        
        step_data['action_reason'] = reason
        step_data['action'] = action.strip()

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
        # Convert the action to JSONAction
        if 'swipe' in action.lower():
            if ' down ' in reason.lower():
                action = '{"action_type": "scroll", "direction": "up"}'
            elif ' up ' in reason.lower():
                action = '{"action_type": "scroll", "direction": "down"}'
            elif ' left ' in reason.lower():
                action = '{"action_type": "scroll", "direction": "right"}'
            elif ' right ' in reason.lower():
                action = '{"action_type": "scroll", "direction": "left"}'
            else:
                action = '{"action_type": "scroll", "direction": "down"}'  # default to down
                
        import traceback
        try:
            converted_action = json_action.JSONAction(
                **agent_utils.extract_json(action),
            )
            step_data['action_output_json'] = converted_action

            converted_action.coordinates = None
            converted_action.element = None
            if 'qwen3' in self.llm.model.lower():
                # x, y / 1000 * image width/height
                if converted_action.x is not None and converted_action.y is not None:
                    original_x, original_y = converted_action.x, converted_action.y
                    converted_action.x = converted_action.x / 1000 * img.width
                    converted_action.y = converted_action.y / 1000 * img.height
                    print('Qwen3 change coordinates from {} to: {}'.format(
                        (original_x, original_y),
                        (converted_action.x, converted_action.y)
                    ))
            print(converted_action)
            # breakpoint()

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

            if converted_action.action_type == 'open_app':
                step_data['summary'] = (
                    'Failed to open the app. Please make sure using the correct app name or choose other ways to open a correct app!'
                )
            else:
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
