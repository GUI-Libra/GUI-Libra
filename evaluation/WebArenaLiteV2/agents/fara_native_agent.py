# Adapted from FARA (https://github.com/ARC-AGI/FARA) for the ScaleCUA-based
# WebArena-Lite-v2 evaluation framework. FARA-7B is a Qwen2.5-VL based model
# that uses <tool_call> JSON output format for web automation.
#
# Key differences from the ScaleCUA native agent:
#   - Tool-call output format (<tool_call>JSON</tool_call>) instead of
#     Python function calls inside <action> tags
#   - Multi-turn conversation with screenshot history
#   - Action space: left_click, type, key, scroll, wait,
#     pause_and_memorize_fact, terminate, web_search, visit, history_back

import ast
import json
import logging
import re
from typing import Dict, List, Tuple

from agents.ui_agent import UIAgent
from utils.misc import call_llm_safe, smart_resize, IMAGE_FACTOR

logger = logging.getLogger()

# FARA's image processing parameters (from Qwen2.5-VL processor config)
FARA_MIN_PIXELS = 3136  # 4 * 28 * 28
FARA_MAX_PIXELS = 12845056  # 16384 * 28 * 28

# Map localhost service ports to the human-readable domain names FARA was trained on.
# This helps the model's instruction-following since it expects real URLs in context.
_LOCALHOST_URL_MAP = {
    "localhost:7770": "www.shopping.com",
    "localhost:7780": "www.shopping_admin.com",
    "localhost:9999": "www.reddit.com",
    "localhost:8023": "www.gitlab.com",
    "localhost:3000": "www.map.com",
}


class FaraNativeAgent(UIAgent):
    """Agent adapter for FARA-7B within the ScaleCUA evaluation framework."""

    USER_MESSAGE = "Here is the next screenshot. Think about what to do next."

    def __init__(
        self,
        engine_params: Dict,
        platform: str = "web",
        width: int = 1280,
        height: int = 720,
    ):
        super().__init__(engine_params=engine_params, platform=platform)

        self.width = width
        self.height = height

        # Compute FARA's resized image dimensions (the model outputs coordinates
        # in this pixel space, so we need it for coordinate conversion).
        self.resized_height, self.resized_width = smart_resize(
            height,
            width,
            factor=IMAGE_FACTOR,
            min_pixels=FARA_MIN_PIXELS,
            max_pixels=FARA_MAX_PIXELS,
        )

        # Build FARA's system prompt with tool descriptions
        self.sys_prompt = self._build_system_prompt()

        # Load user-turn prompt templates
        with open(self.engine_params["prompt_template"], "r", encoding="utf-8") as f:
            self.prompt_template = json.load(f)

        self.messages = None
        self.native_agent = None
        self.step_count = 0
        self.memorized_facts = []
        self.reset()

    # ------------------------------------------------------------------
    # System prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Assemble FARA's full system prompt with embedded tool descriptions."""

        tool_description = (
            "Use a mouse and keyboard to interact with a computer, and take screenshots.\n"
            "* This is an interface to a desktop GUI. You do not have access to a terminal "
            "or applications menu. You must click on desktop icons to start applications.\n"
            "* Some applications may take time to start or process actions, so you may need "
            "to wait and take successive screenshots to see the results of your actions. "
            "E.g. if you click on Firefox and a window doesn't open, try wait and taking "
            "another screenshot.\n"
            f"* The screen's resolution is {self.resized_width}x{self.resized_height}.\n"
            "* Whenever you intend to move the cursor to click on an element like an icon, "
            "you should consult a screenshot to determine the coordinates of the element "
            "before moving the cursor.\n"
            "* If you tried clicking on a program or link but it failed to load, even after "
            "waiting, try adjusting your cursor position so that the tip of the cursor "
            "visually falls on the element that you want to click.\n"
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the "
            "center of the element. Don't click boxes on their edges unless asked.\n"
            "* When a separate scrollable container prominently overlays the webpage, if you "
            "want to scroll within it, you typically need to mouse_move() over it first and "
            "then scroll().\n"
            "* If a popup window appears that you want to close, if left_click() on the 'X' "
            "or close button doesn't work, try key(keys=['Escape']) to close it.\n"
            "* On some search bars, when you type(), you may need to press_enter=False and "
            "instead separately call left_click() on the search button to submit the search "
            "query. This is especially true of search bars that have auto-suggest popups "
            "for e.g. locations\n"
            "* For calendar widgets, you usually need to left_click() on arrows to move "
            "between months and left_click() on dates to select them; type() is not "
            "typically used to input dates there."
        )

        tool_parameters = {
            "properties": {
                "action": {
                    "description": (
                        "The action to perform. The available actions are:\n"
                        '* `key`: Performs key down presses on the arguments passed in order, '
                        'then performs key releases in reverse order. Includes "Enter", "Alt", '
                        '"Shift", "Tab", "Control", "Backspace", "Delete", "Escape", "ArrowUp", '
                        '"ArrowDown", "ArrowLeft", "ArrowRight", "PageDown", "PageUp", "Shift", etc.\n'
                        "* `type`: Type a string of text on the keyboard.\n"
                        "* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n"
                        "* `left_click`: Click the left mouse button.\n"
                        "* `scroll`: Performs a scroll of the mouse scroll wheel.\n"
                        "* `visit_url`: Visit a specified URL.\n"
                        "* `web_search`: Perform a web search with a specified query.\n"
                        "* `history_back`: Go back to the previous page in the browser history.\n"
                        "* `pause_and_memorize_fact`: Pause and memorize a fact for future reference.\n"
                        "* `wait`: Wait specified seconds for the change to happen.\n"
                        "* `terminate`: Terminate the current task and report its completion status."
                    ),
                    "enum": ["key", "type", "mouse_move", "left_click", "scroll", "visit_url", "web_search", "history_back", "pause_and_memorize_fact", "wait", "terminate"],
                    "type": "string",
                },
                "coordinate": {
                    "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=left_click`, `action=mouse_move`, and `action=type`.",
                    "type": "array",
                },
                "text": {
                    "description": "Required only by `action=type`.",
                    "type": "string",
                },
                "press_enter": {
                    "description": "Whether to press the Enter key after typing. Required only by `action=type`.",
                    "type": "boolean",
                },
                "delete_existing_text": {
                    "description": "Whether to delete existing text before typing. Required only by `action=type`.",
                    "type": "boolean",
                },
                "keys": {
                    "description": "Required only by `action=key`.",
                    "type": "array",
                },
                "pixels": {
                    "description": (
                        "The amount of scrolling to perform. Positive values scroll up, "
                        "negative values scroll down. Required only by `action=scroll`."
                    ),
                    "type": "number",
                },
                "time": {
                    "description": "The seconds to wait. Required only by `action=wait`.",
                    "type": "number",
                },
                "fact": {
                    "description": (
                        "The fact to remember for the future. Required only by "
                        "`action=pause_and_memorize_fact`."
                    ),
                    "type": "string",
                },
                "status": {
                    "description": "The status of the task. Required only by `action=terminate`.",
                    "type": "string",
                    "enum": ["success", "failure"],
                },
                "url": {
                    "description": "The URL to visit. Required only by `action=visit_url`.",
                    "type": "string",
                },
                "query": {
                    "description": "The query to search for. Required only by `action=web_search`.",
                    "type": "string",
                },
            },
            "required": ["action"],
            "type": "object",
        }

        tool_json = json.dumps(
            {
                "type": "function",
                "function": {
                    "name": "computer_use",
                    "description": tool_description,
                    "parameters": tool_parameters,
                },
            },
            ensure_ascii=False,
        )

        tool_system = (
            "You are a web automation agent that performs actions on websites to "
            "fulfill user requests by calling various tools.\n"
            "* You should stop execution at Critical Points. A Critical Point would be "
            "encountered in tasks like 'Checkout', 'Book', 'Purchase', 'Call', 'Email', "
            "'Order', etc where a binding transaction/agreement would require the user's "
            "permission/personal or sensitive information (name, email, credit card, "
            "address, payment information, resume, etc) in order to complete a transaction "
            "(purchase, reservation, sign-up etc), or to communicate in a way that a "
            "human would be expected to do (call, email, apply to a job, etc).\n"
            "* Solve the task as far as you can up until a Critical Point:\n"
            '    - For example, if the task is to "call a restaurant to make a reservation", '
            "you should not actually make the call but should navigate to the restaurant's "
            "page and find the phone number.\n"
            '    - Similarly, if the task is to "order new size 12 running shoes" you should '
            "not actually place the order but should instead search for the right shoes that "
            "meet the criteria and add them to the cart.\n"
            "    - Some tasks, like answering questions, may not encounter a Critical Point "
            "at all.\n"
            "\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>\n"
            f"{tool_json}\n"
            "</tools>\n"
            "\n"
            "For each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>"
        )

        return "You are a helpful assistant.\n\n" + tool_system

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        self.messages = [{"role": "system", "content": self.sys_prompt}]
        self.native_agent = self._create_agent("")
        self.step_count = 0
        self.memorized_facts = []

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_tool_call(self, response_text: str):
        """Extract thoughts and tool-call JSON from FARA's response."""
        parts = response_text.split("<tool_call>")
        thoughts = parts[0].strip() if len(parts) > 1 else response_text.strip()

        tool_call_match = re.search(
            r"<tool_call>\s*(.*?)\s*</tool_call>", response_text, re.DOTALL
        )
        if not tool_call_match:
            return thoughts, None

        action_text = tool_call_match.group(1).strip()
        try:
            action = json.loads(action_text)
        except json.JSONDecodeError:
            try:
                action = ast.literal_eval(action_text)
            except (ValueError, SyntaxError):
                logger.error("Failed to parse tool call JSON: %s", action_text)
                return thoughts, None
        return thoughts, action

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def _scale_coordinate(self, coord):
        """Convert FARA's resized-image pixel coordinates to screen pixel coordinates."""
        if coord is None or not isinstance(coord, (list, tuple)) or len(coord) != 2:
            return None, None
        x, y = coord
        screen_x = x * (self.width / self.resized_width)
        screen_y = y * (self.height / self.resized_height)
        return screen_x, screen_y

    # ------------------------------------------------------------------
    # Key-name mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _map_key_name(key: str) -> str:
        """Map FARA's key names (PascalCase) to the framework's lowercase names."""
        key_map = {
            "Enter": "enter",
            "Return": "enter",
            "Backspace": "backspace",
            "Delete": "delete",
            "Escape": "escape",
            "Tab": "tab",
            "Space": "space",
            "ArrowUp": "up",
            "ArrowDown": "down",
            "ArrowLeft": "left",
            "ArrowRight": "right",
            "PageUp": "pageup",
            "PageDown": "pagedown",
            "Home": "home",
            "End": "end",
            "Control": "ctrl",
            "Alt": "alt",
            "Shift": "shift",
            "Meta": "win",
        }
        return key_map.get(key, key.lower())

    # ------------------------------------------------------------------
    # Action conversion  (FARA action space → framework action space)
    # ------------------------------------------------------------------

    def _convert_action(self, tool_call) -> List[dict]:
        """Convert a FARA tool-call dict to a list of framework actions.

        Action space: key, type, mouse_move, left_click, scroll, visit_url,
        web_search, history_back, pause_and_memorize_fact, wait, terminate.
        """
        if tool_call is None:
            return [{"name": "wait", "parameters": {"seconds": 1}}]

        args = tool_call.get("arguments", {})
        action_type = args.get("action", "")

        # --- mouse_move ---
        if action_type == "mouse_move":
            x, y = self._scale_coordinate(args.get("coordinate"))
            return [
                {"name": "moveTo", "parameters": {"x": x, "y": y}}
            ]

        # --- left_click ---
        elif action_type in ("left_click", "click"):
            x, y = self._scale_coordinate(args.get("coordinate"))
            return [
                {"name": "click", "parameters": {"x": x, "y": y, "clicks": 1, "button": "left"}}
            ]

        # --- type / write ---
        elif action_type in ("type", "input_text"):
            exec_actions = []
            coord = args.get("coordinate")
            if coord and isinstance(coord, (list, tuple)) and len(coord) == 2:
                x, y = self._scale_coordinate(coord)
                if x is not None and y is not None:
                    exec_actions.append(
                        {"name": "click", "parameters": {"x": x, "y": y, "clicks": 1, "button": "left"}}
                    )
            # The framework's write() always clears the field first (Ctrl+A + Backspace),
            # so delete_existing_text has no additional effect.
            text = args.get("text", "")
            exec_actions.append({"name": "write", "parameters": {"message": text}})
            # press_enter defaults to True (matches magentic-ui's FARA integration)
            if args.get("press_enter", True):
                exec_actions.append({"name": "press", "parameters": {"keys": "enter"}})
            return exec_actions or [{"name": "wait", "parameters": {"seconds": 1}}]

        # --- key / press ---
        elif action_type in ("key", "keypress"):
            keys = args.get("keys", [])
            if isinstance(keys, list) and len(keys) > 0:
                mapped = [self._map_key_name(k) for k in keys]
                if len(mapped) == 1:
                    return [{"name": "press", "parameters": {"keys": mapped[0]}}]
                else:
                    return [{"name": "hotkey", "parameters": {"args": mapped}}]
            return [{"name": "wait", "parameters": {"seconds": 1}}]

        # --- scroll ---
        elif action_type == "scroll":
            pixels = args.get("pixels", 0)
            # FARA: positive pixels = scroll up (see content above)
            # Framework swipe: direction is the finger-drag direction, which is
            # opposite to scroll direction — swipe "down" scrolls page up.
            # This matches GUILibra's mapping: scroll "up" → swipe "down".
            direction = "down" if pixels >= 0 else "up"
            return [
                {
                    "name": "swipe",
                    "parameters": {
                        "direction": direction,
                        "amount": min(abs(pixels) / 500, 1.0),
                    },
                }
            ]

        # --- wait ---
        elif action_type in ("wait", "sleep"):
            # Accept both "duration" and "time" param names for compat
            seconds = args.get("duration", 3)
            seconds = args.get("time", seconds)
            return [{"name": "wait", "parameters": {"seconds": seconds}}]

        # --- pause_and_memorize_fact ---
        elif action_type == "pause_and_memorize_fact":
            fact = args.get("fact", "")
            if fact:
                self.memorized_facts.append(fact)
            # No env action; just a brief wait so the loop advances
            return [{"name": "wait", "parameters": {"seconds": 1}}]

        # --- terminate ---
        elif action_type in ("terminate", "stop"):
            status = args.get("status", "success")
            # For QA tasks: emit response with the last memorized fact as the answer
            if self.memorized_facts:
                answer = self.memorized_facts[-1]
                return [{"name": "response", "parameters": {"answer": answer}}]
            return [{"name": "terminate", "parameters": {"status": status, "info": ""}}]

        # --- web_search ---
        elif action_type == "web_search":
            query = args.get("query", "")
            return [{"name": "go_to_url", "parameters": {"url": f"https://www.google.com/search?q={query}"}}]

        # --- visit_url / visit ---
        elif action_type in ("visit_url", "visit"):
            url = args.get("url", "")
            return [{"name": "go_to_url", "parameters": {"url": url}}]

        # --- history_back ---
        elif ("history" in action_type) or ("back" in action_type):
            return [{"name": "back", "parameters": {}}]

        else:
            logger.warning("Unknown FARA action type: %s", action_type)
            return [{"name": "wait", "parameters": {"seconds": 1}}]

    # ------------------------------------------------------------------
    # URL normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_url(raw_url: str) -> str:
        """Replace localhost:PORT with the trained domain name and strip query params."""
        url = raw_url.split("?", 1)[0]  # strip query string
        for local, domain in _LOCALHOST_URL_MAP.items():
            if local in url:
                url = url.replace(local, domain)
                break
        if len(url) > 100:
            url = url[:100] + " ..."
        return url

    # ------------------------------------------------------------------
    # Image-history management
    # ------------------------------------------------------------------

    def _limit_images(self, max_images: int = 3):
        """Drop old screenshots from conversation to manage token usage."""
        image_locations = []  # list of (message_index, content_item_index)
        for i, msg in enumerate(self.messages):
            if msg["role"] == "user" and isinstance(msg.get("content"), list):
                for j, item in enumerate(msg["content"]):
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        image_locations.append((i, j))

        # Remove oldest images first, keeping the most recent max_images
        while len(image_locations) > max_images:
            msg_idx, item_idx = image_locations.pop(0)
            content = self.messages[msg_idx]["content"]
            # Replace the image item with a text placeholder
            content[item_idx] = {"type": "text", "text": "[screenshot omitted]"}

    # ------------------------------------------------------------------
    # Main predict loop
    # ------------------------------------------------------------------

    def generate_next_action(self, instruction: str, obs: Dict) -> Tuple[Dict, List]:
        # Encode screenshot
        image_content = obs["screenshot"]
        base64_image, img_size = self.native_agent.encode_image(image_content)

        # Build user message
        if self.step_count == 0:
            user_text = self.prompt_template["user_prompt_first"].format(
                instruction=instruction
            )
        else:
            user_text = self.prompt_template["user_prompt_subsequent"]
            # Prepend current URL (mapped to trained domain names) as FARA expects
            raw_url = obs.get("url", "")
            if raw_url:
                normalized = self._normalize_url(raw_url)
                user_text = f"Current URL: {normalized}\n" + user_text

        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": user_text},
                ],
            }
        )

        # Keep token budget manageable
        self._limit_images(max_images=3)

        # Call LLM
        self.native_agent.replace_messages(self.messages)
        logger.info("Input Messages: %s", self.messages)
        response = call_llm_safe(self.native_agent)
        logger.info("API Response Plan: %s", response)

        # Append assistant turn to conversation history
        self.messages.append({"role": "assistant", "content": response})

        # Parse response
        thoughts, tool_call = self._parse_tool_call(response)

        # Convert to framework actions
        try:
            exec_actions = self._convert_action(tool_call)
        except Exception as e:
            logger.error("Error converting FARA action: %s", e)
            exec_actions = [{"name": "wait", "parameters": {"seconds": 1}}]

        # Build operation summary for logging / result tracking
        operation = ""
        if tool_call and "arguments" in tool_call:
            action_type = tool_call["arguments"].get("action", "")
            operation = f"{action_type}: {json.dumps(tool_call['arguments'], ensure_ascii=False)}"

        self.step_count += 1

        exec_info = {
            "thought": thoughts,
            "operation": operation,
            "actions": json.dumps(tool_call, ensure_ascii=False) if tool_call else "",
        }
        return exec_info, exec_actions

    def predict(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        return self.generate_next_action(instruction, observation)
