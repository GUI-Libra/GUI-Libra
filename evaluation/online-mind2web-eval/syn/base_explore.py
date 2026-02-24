import json
import os
import re
import time
import signal as signal_mod
import threading
import numpy as np
from tqdm import tqdm
from simpleArgParser import to_json


class PlaywrightTimeoutError(Exception):
    pass


def _get_browser_pids(env) -> dict:
    result = {'driver_pid': None, 'chrome_pids': []}
    try:
        browser = env.browser if hasattr(env, 'browser') else env.context.browser
        transport = browser._impl_obj._connection._transport
        if hasattr(transport, '_proc') and transport._proc is not None:
            driver_pid = transport._proc.pid
            result['driver_pid'] = driver_pid
            import subprocess
            try:
                pgrep_result = subprocess.run(
                    ['pgrep', '-P', str(driver_pid)],
                    capture_output=True, text=True, timeout=5
                )
                result['chrome_pids'] = [int(p) for p in pgrep_result.stdout.strip().split('\n') if p.strip()]
            except Exception:
                pass
    except Exception:
        pass
    return result


class BrowserWatchdog:
    FORCE_EXIT_GRACE = 10
    FORCE_EXIT_CODE = 99

    def __init__(self, timeout_seconds: int, description: str = ""):
        self.timeout = timeout_seconds
        self.description = description
        self._chrome_timer: threading.Timer | None = None
        self._exit_timer: threading.Timer | None = None
        self._pids: dict = {}
        self._fired = False

    def start(self, env) -> "BrowserWatchdog":
        self._pids = _get_browser_pids(env)
        self._fired = False
        self._chrome_timer = threading.Timer(self.timeout, self._kill_chrome)
        self._chrome_timer.daemon = True
        self._chrome_timer.start()
        total_timeout = self.timeout + self.FORCE_EXIT_GRACE
        self._exit_timer = threading.Timer(total_timeout, self._force_exit)
        self._exit_timer.daemon = True
        self._exit_timer.start()
        return self

    def cancel(self):
        if self._chrome_timer is not None:
            self._chrome_timer.cancel()
            self._chrome_timer = None
        if self._exit_timer is not None:
            self._exit_timer.cancel()
            self._exit_timer = None

    @property
    def did_fire(self) -> bool:
        return self._fired

    def _kill_chrome(self):
        self._fired = True
        chrome_pids = self._pids.get('chrome_pids', [])
        driver_pid = self._pids.get('driver_pid')
        if chrome_pids:
            logger.warning(f"BrowserWatchdog: '{self.description}' exceeded {self.timeout}s. Killing Chrome PIDs {chrome_pids}.")
            for pid in chrome_pids:
                try:
                    os.kill(pid, signal_mod.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        elif driver_pid:
            logger.warning(f"BrowserWatchdog: '{self.description}' exceeded {self.timeout}s. Killing driver PID {driver_pid}.")
            try:
                os.kill(driver_pid, signal_mod.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def _force_exit(self):
        logger.error(f"BrowserWatchdog: '{self.description}' still stuck. Force-exiting with os._exit({self.FORCE_EXIT_CODE}).")
        os._exit(self.FORCE_EXIT_CODE)


class step_timeout:
    def __init__(self, seconds: int, description: str = "", env=None):
        self.seconds = seconds
        self.description = description
        self.env = env
        self._watchdog: BrowserWatchdog | None = None

    def __enter__(self):
        if self.env is not None:
            self._watchdog = BrowserWatchdog(self.seconds, self.description)
            self._watchdog.start(self.env)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._watchdog is not None:
            self._watchdog.cancel()
        return False


from syn.gpt import GPTClient
from syn.data import (
    Element, LowLevelTask, HighLevelTask, LowTaskStatus, Action, StateInfo,
    ActionType, RawState,
)
from syn.args import AgentConfig
from syn.tools import (
    tools_get_time,
)
from loguru import logger


class Explorer:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.gpt_client = GPTClient(
            provider=self.config.gpt.provider,
            base_url=self.config.gpt.openai_api_base,
            api_key=self.config.gpt.openai_api_key,
        )
        self._env: None | "ScriptBrowserEnv" = None


    def save(self):
        with open(f"{self.config.output}/config.json", 'w') as f:
            f.write(to_json(self.config))
        self.gpt_client.token_usage.to_json(f"{self.config.output}/gpt_client_token_usage.json")


    def load(self):
        if os.path.exists(path := f"{self.config.output}/gpt_client_token_usage.json"):
            self.gpt_client.token_usage.load_from_json(path)


    def goto_url(self, env: "ScriptBrowserEnv", curr_state: StateInfo, url: str) -> StateInfo:
        from browser_env import create_goto_url_action
        obs, _, _, _, info = env.step(create_goto_url_action(url))
        time.sleep(3)
        info_meta = info['observation_metadata']
        return self._get_env_state(env, obs, info_meta)


    def extract_elements(self, raw_state: RawState) -> list[Element]:
        elements = [
            Element(
                accessibility_tree_content=ele['text'],
                union_bound=ele['union_bound'],
                element_id=ele_id
            )
            for ele_id, ele in raw_state.observation_metadata['text']['obs_nodes_info'].items()
        ]
        return elements


    def _check_browser_alive(self, env: "ScriptBrowserEnv") -> bool:
        try:
            with step_timeout(15, "browser liveness check", env=env):
                _ = env.page.url
            return True
        except Exception as e:
            logger.warning(f"Browser liveness check failed: {e}")
            return False

    def _recover_browser(self, env: "ScriptBrowserEnv", start_url: str) -> tuple["ScriptBrowserEnv", StateInfo]:
        logger.warning(f"Attempting browser recovery...")
        pids = _get_browser_pids(env)
        driver_pid = pids.get('driver_pid')
        chrome_pids = pids.get('chrome_pids', [])
        for pid in chrome_pids:
            try:
                os.kill(pid, signal_mod.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        if driver_pid:
            try:
                os.kill(driver_pid, signal_mod.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

        def _close_env():
            try:
                env.close()
            except Exception:
                pass

        close_thread = threading.Thread(target=_close_env, daemon=True)
        close_thread.start()
        close_thread.join(timeout=10)
        time.sleep(5.0)
        new_env, new_state = self._init_env_for_episode(start_url)
        logger.info(f"Browser recovery successful at {start_url}")
        return new_env, new_state

    def _init_env_for_episode(self, start_url: str) -> tuple["ScriptBrowserEnv", StateInfo]:
        from browser_env import ScriptBrowserEnv
        env = ScriptBrowserEnv(
            headless=self.config.browser.headless,
            slow_mo=self.config.browser.slow_mo,
            observation_type=self.config.browser.observation_type,
            current_viewport_only=self.config.browser.current_viewport_only,
            viewport_size=self.config.browser.viewport_size,
            sleep_after_execution=self.config.sleep_after_action,
        )
        self._env = env
        observation, info = self._reset_env(env, start_url=start_url)
        observation_metadata = info['observation_metadata']
        current_state = self._get_env_state(env, obs=observation, observation_metadata=observation_metadata)
        return env, current_state

    def _reset_all_tabs_and_open_seed_url(self, env, seed_url: str) -> StateInfo:
        from browser_env import create_page_close_action, create_new_tab_action, create_goto_url_action, create_page_focus_action
        try:
            num_tabs = len(env.context.pages)
            for i in range(num_tabs - 1, 0, -1):
                try:
                    env.step(create_page_focus_action(i))
                    env.step(create_page_close_action())
                except Exception as e:
                    logger.warning(f"Failed to close tab {i}: {e}")
            obs, _, _, _, info = env.step(create_goto_url_action(seed_url))
            info_meta = info['observation_metadata']
            return self._get_env_state(env, obs, info_meta)
        except Exception as e:
            logger.error(f"Error resetting tabs: {e}")
            from browser_env import create_goto_url_action
            obs, _, _, _, info = env.step(create_goto_url_action(seed_url))
            info_meta = info['observation_metadata']
            return self._get_env_state(env, obs, info_meta)

    def _reset_env(self, env: "ScriptBrowserEnv", start_url: str):
        with open(f"{self.config.output}/init_env.json", 'w') as f:
            state = {'start_url': start_url, 'storage_state': None}
            json.dump(state, f)
            logger.info(f"Resetting environment with state: {state}")
        observation, info = env.reset(options={'config_file': f"{self.config.output}/init_env.json"})
        env.context.set_default_timeout(30000)
        env.context.set_default_navigation_timeout(60000)
        return observation, info

    SCALECUA_KEY_MAPPING = {
        "accept": "Accept", "add": "Add", "alt": "Alt", "altleft": "AltLeft",
        "altright": "AltRight", "apps": "Apps", "backspace": "Backspace",
        "browserback": "BrowserBack", "browserfavorites": "BrowserFavorites",
        "browserforward": "BrowserForward", "browserhome": "BrowserHome",
        "browserrefresh": "BrowserRefresh", "browsersearch": "BrowserSearch",
        "browserstop": "BrowserStop", "capslock": "CapsLock", "clear": "Clear",
        "convert": "Convert", "ctrl": "Control", "ctrlleft": "ControlLeft",
        "ctrlright": "ControlRight", "control": "Control",
        "decimal": "Decimal", "del": "Delete", "delete": "Delete", "divide": "Divide",
        "down": "ArrowDown", "end": "End", "enter": "Enter",
        "esc": "Escape", "escape": "Escape", "execute": "Execute",
        "f1": "F1", "f2": "F2", "f3": "F3", "f4": "F4", "f5": "F5",
        "f6": "F6", "f7": "F7", "f8": "F8", "f9": "F9", "f10": "F10",
        "f11": "F11", "f12": "F12",
        "final": "Final", "fn": "Fn", "help": "Help", "home": "Home",
        "insert": "Insert", "left": "ArrowLeft",
        "modechange": "ModeChange", "multiply": "Multiply",
        "nexttrack": "MediaTrackNext", "nonconvert": "NonConvert",
        "num0": "Numpad0", "num1": "Numpad1", "num2": "Numpad2",
        "num3": "Numpad3", "num4": "Numpad4", "num5": "Numpad5",
        "num6": "Numpad6", "num7": "Numpad7", "num8": "Numpad8",
        "num9": "Numpad9", "numlock": "NumLock",
        "pagedown": "PageDown", "pageup": "PageUp", "pause": "Pause",
        "pgdn": "PageDown", "pgup": "PageUp",
        "playpause": "MediaPlayPause", "prevtrack": "MediaTrackPrevious",
        "print": "Print", "printscreen": "PrintScreen", "prntscrn": "PrintScreen",
        "prtsc": "PrintScreen", "prtscr": "PrintScreen",
        "return": "Enter", "right": "ArrowRight",
        "scrolllock": "ScrollLock", "select": "Select", "separator": "Separator",
        "shift": "Shift", "shiftleft": "ShiftLeft", "shiftright": "ShiftRight",
        "sleep": "Sleep", "space": "Space", "stop": "MediaStop",
        "subtract": "Subtract", "tab": "Tab", "up": "ArrowUp",
        "volumedown": "VolumeDown", "volumemute": "VolumeMute",
        "volumeup": "VolumeUp", "win": "Meta", "winleft": "MetaLeft",
        "winright": "MetaRight", "yen": "Yen",
        "command": "Meta", "cmd": "Meta", "meta": "Meta",
        "option": "Alt", "optionleft": "AltLeft", "optionright": "AltRight",
        "arrowup": "ArrowUp", "arrowdown": "ArrowDown",
        "arrowleft": "ArrowLeft", "arrowright": "ArrowRight",
    }

    STEP_TIMEOUT = 120


    def _execute_single_low_level_task(self, task: LowLevelTask, env: "ScriptBrowserEnv", curr_state: StateInfo | None = None) -> StateInfo:
        from browser_env import create_id_based_action, create_none_action

        env.page.set_default_navigation_timeout(60000)
        env.page.set_default_timeout(30000)

        if task.state_after is not None:
            if task.state_after.raw_state.url == env.page.url:
                return task.state_after
            else:
                logger.warning(f"state_after URL mismatch, re-executing task.")
        else:
            if curr_state is None:
                curr_state = self._get_env_state(env, None, None)

        for page in env.context.pages:
            page.set_default_navigation_timeout(60000)
            page.set_default_timeout(30000)

        if task.action.action_type == ActionType.NONE:
            obs = env._get_obs()
            observation_metadata = env._get_obs_metadata()
            info = {'observation_metadata': observation_metadata}
        else:
            exec_actions = getattr(task.action, 'exec_actions', None)
            if exec_actions or (task.action.coordinates and task.action.target_element is None):
                obs, info = self._execute_coordinate_based_action(task.action, env)
            else:
                obs, _, _, _, info = env.step(create_id_based_action(task.action.get_action_str()))
            time.sleep(self.config.sleep_after_action)

        logger.info(f"Executed action_str={task.action.get_action_str()}, coordinates={task.action.coordinates}")

        new_state = self._get_env_state(env, obs=obs, observation_metadata=info['observation_metadata'])
        task.state_after = new_state
        return new_state

    def _execute_coordinate_based_action(self, action: "Action", env: "ScriptBrowserEnv") -> tuple[dict, dict]:
        page = env.page
        exec_actions = getattr(action, 'exec_actions', None)

        if exec_actions:
            for exec_action in exec_actions:
                name = exec_action.get("name", "")
                params = exec_action.get("parameters", {})
                logger.info(f"Executing action: {name} with params={params}")
                try:
                    if name == "click":
                        x, y = params.get("x"), params.get("y")
                        clicks = params.get("clicks", 1)
                        button = params.get("button", "left")
                        if x is not None and y is not None:
                            page.mouse.click(x, y, button=button, click_count=clicks)
                        time.sleep(2.0)
                    elif name == "write":
                        message = params.get("message", "")
                        if message:
                            page.keyboard.type(message)
                        time.sleep(1.0)
                    elif name == "press":
                        keys = params.get("keys", "")
                        presses = params.get("presses", 1)
                        if keys:
                            if isinstance(keys, str):
                                mapped_key = self.SCALECUA_KEY_MAPPING.get(keys.lower(), keys)
                                for _ in range(presses):
                                    page.keyboard.press(mapped_key)
                            elif isinstance(keys, list):
                                for key in keys:
                                    mapped_key = self.SCALECUA_KEY_MAPPING.get(str(key).lower(), str(key))
                                    for _ in range(presses):
                                        page.keyboard.press(mapped_key)
                        time.sleep(1.0)
                    elif name == "swipe":
                        direction = params.get("direction", "up")
                        amount = params.get("amount", 0.5)
                        amount = max(0.0, min(1.0, amount))
                        viewport = page.viewport_size
                        vp_width = viewport["width"] if viewport else 1280
                        vp_height = viewport["height"] if viewport else 1000
                        if direction in ["up", "down"]:
                            distance = vp_height * amount if direction == "up" else -vp_height * amount
                            page.evaluate(f"window.scrollBy(0, {distance});")
                        else:
                            distance = vp_width * amount if direction == "left" else -vp_width * amount
                            page.evaluate(f"window.scrollBy({distance}, 0);")
                        time.sleep(1.0)
                    elif name == "back":
                        page.go_back()
                        time.sleep(1.0)
                    elif name == "wait":
                        seconds = params.get("seconds", 3)
                        time.sleep(seconds)
                    elif name in ["response", "answer", "terminate", "call_user", "callUser"]:
                        pass
                    else:
                        logger.warning(f"Unknown exec_action name: {name}")
                except PlaywrightTimeoutError:
                    raise
                except Exception as e:
                    error_msg = str(e).lower()
                    if any(kw in error_msg for kw in ['connection', 'closed', 'broken pipe', 'target page', 'browser has been closed', 'protocol error']):
                        raise
                    logger.error(f"Error executing action {name}: {e}")
        else:
            if action.coordinates:
                x, y = action.coordinates
                if action.action_type == ActionType.CLICK:
                    page.mouse.click(x, y)
                elif action.action_type == ActionType.TYPE:
                    page.mouse.click(x, y)
                    time.sleep(0.5)
                    text = action.value.strip() if action.value else ""
                    if text:
                        page.keyboard.type(text)
                        time.sleep(0.3)
                        page.keyboard.press("Enter")
                elif action.action_type == ActionType.HOVER:
                    page.mouse.move(x, y)
            elif action.action_type == ActionType.SCROLL:
                direction = action.value.lower() if action.value else "down"
                scroll_amount = 500
                if direction == "up":
                    page.mouse.wheel(0, -scroll_amount)
                elif direction == "down":
                    page.mouse.wheel(0, scroll_amount)
            elif action.action_type == ActionType.PRESS:
                key = action.value.strip() if action.value else ""
                if key:
                    mapped = self.SCALECUA_KEY_MAPPING.get(key.lower(), key)
                    page.keyboard.press(mapped)
            elif action.action_type == ActionType.GO_BACK:
                page.go_back()
                time.sleep(0.5)

        # Switch to the newest tab if a new one was opened (e.g. target="_blank" links)
        current_pages = env.context.pages
        if len(current_pages) > 0 and current_pages[-1] != env.page:
            new_page = current_pages[-1]
            logger.info(f"New tab detected: switching env.page from {env.page.url} to {new_page.url}")
            new_page.bring_to_front()
            if not hasattr(new_page, 'client'):
                new_page.client = new_page.context.new_cdp_session(new_page)
            env.page = new_page

        obs = env._get_obs()
        observation_metadata = env._get_obs_metadata()
        info = {'observation_metadata': observation_metadata}
        return obs, info


    def _get_env_state(self, env: "ScriptBrowserEnv", obs: dict | None, observation_metadata: dict | None) -> StateInfo:
        if obs and observation_metadata:
            pass
        else:
            obs = env._get_obs()
            observation_metadata = env._get_obs_metadata()
        raw_state = RawState(
            url=env.page.url,
            accessibility_tree=obs['text'],
            observation_metadata=observation_metadata,
            screenshot=obs['image'],
            timestamp=time.time(),
        )
        elements = self.extract_elements(raw_state)
        return StateInfo(raw_state=raw_state, elements=elements)

    def _states_different(self, state1: StateInfo, state2: StateInfo) -> bool:
        return (state1.raw_state.url != state2.raw_state.url) or \
               (state1.raw_state.accessibility_tree != state2.raw_state.accessibility_tree) or \
               (not np.array_equal(state1.raw_state.screenshot, state2.raw_state.screenshot))
