from loguru import logger
from simpleArgParser import parse_args
from dataclasses import dataclass, field
from enum import Enum
import os
import re


class APIProvider(Enum):
    openai = "openai"


@dataclass
class BrowserConfig:
    headless: bool = True
    slow_mo: int = 0
    observation_type: str = "accessibility_tree"
    viewport_size = {"width": 1280, "height": 1000}
    current_viewport_only: bool = True


@dataclass
class GPTConfig:
    model: str = "gpt-4.1"
    temperature: float = 0.7
    max_completion_tokens: int = 4096
    provider: APIProvider = APIProvider.openai
    openai_api_base: str = "https://api.openai.com/v1"
    openai_api_key: str = "token-abc123"

    def post_process(self):
        if self.model.startswith('o'):
            self.temperature = 1.0
            logger.warning(f"Model {self.model} is o series, setting temperature to 1.0")


@dataclass
class DebugConfig:
    debug: bool = False


def get_model_short_name(model: str) -> str:
    name = os.path.basename(model)
    name = re.sub(r'[-_]Instruct$', '', name, flags=re.IGNORECASE)
    name = name.lower()
    name = re.sub(r'[_\s]+', '-', name)
    return name


def is_qwen25_model(model: str) -> bool:
    short = get_model_short_name(model)
    return 'qwen2.5' in short or 'qwen2.5vl' in short


@dataclass
class AgentConfig:
    tasks_path: str = field(kw_only=True)
    browser: BrowserConfig = field(default_factory=BrowserConfig)
    gpt: GPTConfig = field(default_factory=GPTConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    max_steps: int = 30
    history_last_k: int = 15
    sleep_after_action: float = 2.0
    failed_retry: int = 2

    output: str | None = None
    note: str | None = None
    limit: int | None = None
    external_grounding_port: int | None = None

    def pre_process(self):
        assert os.path.exists(self.tasks_path), f"Tasks path={self.tasks_path} does not exist"

        from syn.tools import tools_get_time
        from syn.data import set_screenshot_save_path

        if self.output is None:
            tasks_name = os.path.splitext(os.path.basename(self.tasks_path))[0]
            model_name = get_model_short_name(self.gpt.model)
            self.output = f"outputs/{tasks_name}.{model_name}.{tools_get_time()}"
            if isinstance(self.note, str):
                self.output += f"_{self.note}"

        if self.debug.debug and not self.output.startswith('outputs/debug/'):
            self.output = self.output.replace('outputs/', 'outputs/debug/')

        os.makedirs(self.output, exist_ok=True)

        log_file = f"{self.output}/run.log"
        logger.add(
            log_file,
            format='<green>{time:YY-MM-DD HH:mm:ss.SS}</green> | <level>{level: <5}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>\n<level>{message}</level>',
            level='DEBUG',
            colorize=False,
            rotation=None,
        )

        screenshots_path = f"{self.output}/screenshots"
        os.makedirs(screenshots_path, exist_ok=True)
        set_screenshot_save_path(screenshots_path)

        self.sleep_after_action = max(self.sleep_after_action, 2.0)

        if self.external_grounding_port is None and 'gpt' in self.gpt.model.lower():
            self.external_grounding_port = 9966
            logger.info(f"Auto-enabled external grounding on port {self.external_grounding_port} for GPT model {self.gpt.model}")
