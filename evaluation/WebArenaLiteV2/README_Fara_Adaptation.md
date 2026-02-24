# FARA-7B Adaptation for WebArena-Lite-v2

This document describes how FARA-7B was integrated into the ScaleCUA-based WebArena-Lite-v2 evaluation framework.

FARA-7B is a Qwen2.5-VL based web automation agent that uses a **tool-call output format** (`<tool_call>JSON</tool_call>`), which differs significantly from the ScaleCUA/Qwen native agents that use Python function calls inside `<action>` tags. The adaptation bridges these two formats while preserving FARA's original prompting strategy.

## Files Added / Modified

```
agents/fara_native_agent.py          # NEW  - agent class
config/agent/fara_native_agent.yaml  # NEW  - vLLM + agent config
config/prompt_template/fara.json     # NEW  - user-turn prompt templates
agent_run.py                         # MOD  - registered FaraNativeAgent via agent_type dispatch
```

## What Changed and Why

### 1. System Prompt (`_build_system_prompt`)

FARA expects a specific system prompt structure, reproduced faithfully in `fara_native_agent.py`:

```
You are a helpful assistant.

You are a web automation agent that performs actions on websites...
<tools>
{"type": "function", "function": {"name": "computer_use", ...}}
</tools>

For each function call, return a json object ... within <tool_call></tool_call> XML tags:
```

The tool description embeds the **resized display resolution** (e.g. `1288x728`), computed at init time via Qwen2.5-VL's `smart_resize` with FARA's processor parameters (`min_pixels=3136`, `max_pixels=12845056`). This tells the model the coordinate space its outputs should target.

The existing agents load the system prompt as a static string from a JSON template. Because FARA's prompt must embed dynamic display dimensions and a serialized JSON tool schema, the system prompt is instead **constructed programmatically** in `_build_system_prompt()`. Only the user-turn templates remain in `fara.json`.

### 2. Multi-Turn Conversation

The ScaleCUA native agent uses a **single-turn** approach: it replaces the user message each step, encoding previous operations as text. FARA was trained with **multi-turn** conversations:

```
system:    tool-call prompt
user:      [screenshot_1, task instruction]
assistant: thoughts + <tool_call>...</tool_call>
user:      [screenshot_2, "Here is the next screenshot. Think about what to do next."]
assistant: thoughts + <tool_call>...</tool_call>
...
```

The adapter preserves this by growing `self.messages` across steps and appending each assistant response back into the history.

To manage token usage, `_limit_images(max_images=3)` replaces old screenshot base64 payloads with `[screenshot omitted]` text, keeping only the 3 most recent images. This mirrors FARA's original `max_n_images` budget.

### 3. Response Parsing (`_parse_tool_call`)

FARA outputs free-form reasoning followed by a structured tool call:

```
I need to click the search button...
<tool_call>
{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [640, 350]}}
</tool_call>
```

The parser splits on `<tool_call>` to extract thoughts, then uses regex to extract the JSON between `<tool_call>` and `</tool_call>`. It tries `json.loads` first with `ast.literal_eval` as fallback.

### 4. Constrained Action Space (`_convert_action`)

The adaptation uses a **constrained action space** matching the framework's native capabilities. The browser is pre-navigated to the task's starting URL, so actions like `visit_url`, `web_search`, and `history_back` are removed. Only 6 actions are exposed to the model:

| FARA action | Framework action(s) | Notes |
|---|---|---|
| `left_click(coordinate=[x,y])` | `click(x, y)` | Coordinates scaled from resized to screen space |
| `type(text, coordinate)` | `click(x, y)` + `write(message)` + `press(enter)` | Optional click-to-focus; `press_enter` defaults to **True** |
| `key(keys=[...])` | `press(key)` or `hotkey(keys...)` | Single key vs. key combo |
| `scroll(pixels=N)` | `swipe(direction, amount)` | Positive = scroll up = swipe down; amount proportional to pixel count |
| `wait(time=N)` | `wait(seconds=N)` | Also accepts `duration` param |
| `pause_and_memorize_fact(fact)` | `wait(1)` + stores fact internally | Preserves FARA's original QA pattern |
| `terminate(status)` | `response(answer=last_fact)` if facts memorized, else `terminate(status)` | See QA handling below |

#### `type` Parameters

- **`press_enter`** (boolean): Whether to press Enter after typing. Defaults to **True** (matching the Magentic-UI FARA integration). Set `press_enter=False` for search bars with auto-suggest.
- **`delete_existing_text`** (boolean): The framework's `write()` already clears the field via `Ctrl+A` + `Backspace`, so this flag has no additional effect.

#### `wait` Duration

Accepts both `time` and `duration` parameter names for Magentic-UI compatibility. Default is 3 seconds.

#### Action Aliases (not in schema, handled silently)

| Accepted alias | Canonical action |
|---|---|
| `click` | `left_click` |
| `input_text` | `type` |
| `keypress` | `key` |
| `sleep` | `wait` |
| `stop` | `terminate` |

#### `pause_and_memorize_fact`

This action is kept in the schema to preserve FARA's original QA pattern. The model calls it mid-task to record findings (e.g. a price, a name, a count), then terminates. The adapter accumulates all facts in `self.memorized_facts` and emits the last one as the `response` answer at termination.

### 5. Coordinate Conversion (`_scale_coordinate`)

FARA outputs pixel coordinates relative to the **resized image** (the dimensions reported in the system prompt). The framework expects coordinates in the **original screen pixel space**. The conversion:

```
screen_x = fara_x * (screen_width  / resized_width)
screen_y = fara_y * (screen_height / resized_height)
```

The framework's `WebEnv` then divides by DPR to get CSS viewport coordinates for Playwright.

### 6. Key Name Mapping (`_map_key_name`)

FARA uses PascalCase key names from Playwright (e.g. `"Enter"`, `"ArrowDown"`, `"Control"`). The framework's `KEYBOARD_KEYS` set and `key_mapping` dict expect lowercase (e.g. `"enter"`, `"down"`, `"ctrl"`). A static mapping table handles the translation.

### 7. QA Task Handling

FARA's original QA pattern is preserved: the model uses `pause_and_memorize_fact(fact="the answer")` to store findings during browsing, then calls `terminate(status="success")` when done. The adapter intercepts this at termination: if any facts have been memorized, it emits `response(answer=last_memorized_fact)` instead of `terminate`, which the evaluation framework uses to score QA tasks.

### 8. Agent Dispatch (`agent_run.py`)

A new `agent_type` field in the YAML config selects `FaraNativeAgent`:

```python
agent_type = agent_config["model_config"].get("agent_type", "")
if agent_type == "fara":
    agent = FaraNativeAgent(...)
elif agent_config["model_config"]["model"] == "guilibra":
    agent = GUILibraNativeAgent(...)
else:
    agent = OpenCUANativeAgent(...)
```

This avoids overloading the `model` field and makes it easy to add more agent types in the future.

## Running FARA-7B Evaluation

```bash
# 1. Serve FARA-7B via vLLM
python -m vllm.entrypoints.openai.api_server \
    --served-model-name fara \
    --model microsoft/Fara-7B \
    --port 10028 -tp 2

# 2. Reinitialize the web environment
python launcher/start.py

# 3. Run evaluation
export OPENAI_API_KEY="placeholder"
export OPENAI_BASE_URL="http://localhost:10028/v1"

python agent_run.py \
    --env_config_path config/env/web.yaml \
    --agent_config_path config/agent/fara_native_agent.yaml \
    --task_config_path tasks/ \
    --num_workers 1 \
    --exp_name fara_7b_eval \
    --max_steps 15
```
