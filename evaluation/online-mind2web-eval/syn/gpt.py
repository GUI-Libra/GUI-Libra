import os
from typing import List, Dict, Any, Optional
import asyncio
import concurrent.futures
from openai import AsyncAzureOpenAI, BadRequestError, AsyncOpenAI, InternalServerError
from openai.types.chat import ChatCompletion
from loguru import logger
from tqdm.asyncio import tqdm
import time
from syn.args import APIProvider
from dataclasses import dataclass, field
import httpx


class ChatCompletionFallback(dict):
    def __init__(self, data: Dict[str, Any]):
        super().__init__()
        for k, v in data.items():
            self[k] = self._wrap(v)

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        if name == "message" and "choices" in self and isinstance(self["choices"], list):
            first_choice = self["choices"][0]
            return first_choice.message
        return super().__getattribute__(name)

    @classmethod
    def _wrap(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            return cls(obj)
        if isinstance(obj, list):
            return [cls._wrap(o) for o in obj]
        return obj


@dataclass
class BasicTokenUsage():
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def estimate_cost(self, model: str) -> float:
        cost_per_million_tokens = {
            'gpt-4.1-mini': {'prompt': 0.4, 'completion': 1.6},
            'gpt-4.1': {'prompt': 2, 'completion': 8},
            'gpt-4o': {'prompt': 2.5, 'completion': 10},
            'gpt-4o-mini': {'prompt': 0.15, 'completion': 0.6},
            'o4-mini': {'prompt': 1.1, 'completion': 4.4},
        }
        for short, cost in cost_per_million_tokens.items():
            if short in model:
                prompt_cost = (self.prompt_tokens / 1_000_000) * cost['prompt']
                completion_cost = (self.completion_tokens / 1_000_000) * cost['completion']
                return prompt_cost + completion_cost
        return 0


@dataclass
class TokenUsage():
    usage_by_model: dict[str, BasicTokenUsage] = field(default_factory=lambda: dict(total=BasicTokenUsage()))
    iteration_count: int = 0
    call_num: int = 0

    def stat_token_usage(self, raw: ChatCompletion):
        model = raw.model
        if model not in self.usage_by_model:
            self.usage_by_model[model] = BasicTokenUsage()
        self.usage_by_model[model].prompt_tokens += raw.usage.prompt_tokens
        self.usage_by_model[model].completion_tokens += raw.usage.completion_tokens
        self.usage_by_model[model].total_tokens += raw.usage.total_tokens
        self.usage_by_model['total'].prompt_tokens += raw.usage.prompt_tokens
        self.usage_by_model['total'].completion_tokens += raw.usage.completion_tokens
        self.usage_by_model['total'].total_tokens += raw.usage.total_tokens
        self.call_num += 1

    def __str__(self):
        if not self.usage_by_model or (len(self.usage_by_model) == 1 and 'total' in self.usage_by_model):
            return "No token usage recorded"
        model_width = max(20, max(len(model) for model in self.usage_by_model.keys()) + 2)
        prompt_width = max(15, len("prompt_tokens") + 2)
        completion_width = max(18, len("completion_tokens") + 2)
        total_width = max(13, len("total_tokens") + 2)
        cost_width = max(10, len("cost($)") + 2)
        s = f"{'Model':<{model_width}} | {'prompt_tokens':>{prompt_width}} | {'completion_tokens':>{completion_width}} | {'total_tokens':>{total_width}} | {'cost($)':>{cost_width}}\n"
        s += "-" * (model_width + prompt_width + completion_width + total_width + cost_width + 13) + "\n"
        for model, usage in self.usage_by_model.items():
            if model == 'total':
                continue
            cost = usage.estimate_cost(model)
            s += f"{model:<{model_width}} | {usage.prompt_tokens:>{prompt_width}} | {usage.completion_tokens:>{completion_width}} | {usage.total_tokens:>{total_width}} | {cost:>{cost_width}.2f}\n"
        model = 'total'
        usage = self.usage_by_model[model]
        total_cost = sum(self.usage_by_model[m].estimate_cost(m) for m in self.usage_by_model.keys() if m != 'total')
        s += f"{model:<{model_width}} | {usage.prompt_tokens:>{prompt_width}} | {usage.completion_tokens:>{completion_width}} | {usage.total_tokens:>{total_width}} | {total_cost:>{cost_width}.2f}\n"
        return s

    def per_iteration_str(self, iteration: int | None = None) -> str:
        if iteration is None:
            iteration = self.iteration_count
        iteration = max(1, iteration)
        if not self.usage_by_model or (len(self.usage_by_model) == 1 and 'total' in self.usage_by_model):
            return "No token usage recorded"
        model_width = max(20, max(len(model) for model in self.usage_by_model.keys()) + 2)
        prompt_width = max(15, len("prompt_tokens") + 2)
        completion_width = max(18, len("completion_tokens") + 2)
        total_width = max(13, len("total_tokens") + 2)
        cost_width = max(10, len("cost($)") + 2)
        s = f"{'Model':<{model_width}} | {f'prompt_tokens/{iteration}':>{prompt_width}} | {f'completion_tokens/{iteration}':>{completion_width}} | {f'total_tokens/{iteration}':>{total_width}} | {f'cost($)/{iteration}':>{cost_width}}\n"
        s += "-" * (model_width + prompt_width + completion_width + total_width + cost_width + 13) + "\n"
        for model, usage in self.usage_by_model.items():
            if model == 'total':
                continue
            cost_per_iteration = usage.estimate_cost(model) / iteration
            s += f"{model:<{model_width}} | {usage.prompt_tokens / iteration:>{prompt_width}.2f} | {usage.completion_tokens / iteration:>{completion_width}.2f} | {usage.total_tokens / iteration:>{total_width}.2f} | {cost_per_iteration:>{cost_width}.2f}\n"
        model = 'total'
        usage = self.usage_by_model[model]
        total_cost_per_iteration = sum(self.usage_by_model[m].estimate_cost(m) for m in self.usage_by_model.keys() if m != 'total') / iteration
        s += f"{model:<{model_width}} | {usage.prompt_tokens / iteration:>{prompt_width}.2f} | {usage.completion_tokens / iteration:>{completion_width}.2f} | {usage.total_tokens / iteration:>{total_width}.2f} | {total_cost_per_iteration:>{cost_width}.2f}\n"
        return s

    def to_json(self, path: str):
        import json
        data = {model: usage.__dict__ for model, usage in self.usage_by_model.items()}
        data = {'count': self.iteration_count, 'call': self.call_num, 'usage': data}
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

    def load_from_json(self, path: str):
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        self.usage_by_model = {model: BasicTokenUsage(**usage) for model, usage in data['usage'].items()}
        self.iteration_count = data['count']
        self.call_num = data['call']
        if 'total' not in self.usage_by_model:
            self.usage_by_model['total'] = BasicTokenUsage()


class GPTClient:
    def __init__(self, provider: APIProvider = APIProvider.openai, api_key: Optional[str] = None, base_url: Optional[str] = None):
        match provider:
            case APIProvider.openai:
                http_client = httpx.AsyncClient(verify=False)
                resolved_api_key = api_key or os.environ.get('OPENAI_API_KEY') or 'dummy'
                resolved_base_url = base_url or os.environ.get('OPENAI_API_BASE')
                logger.info(f"Initializing GPTClient with base_url={resolved_base_url}")
                self._client = AsyncOpenAI(
                    api_key=resolved_api_key,
                    base_url=resolved_base_url,
                    http_client=http_client
                )
            case _:
                raise ValueError(f"Unsupported API provider: {provider}")

        self.token_usage = TokenUsage()
        self._max_retry_num = 3
        self._retry_delay_seconds = 1

    def _obj_to_plain(self, obj: Any) -> Any:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (list, tuple)):
            return [self._obj_to_plain(o) for o in obj]
        if isinstance(obj, dict):
            return {k: self._obj_to_plain(v) for k, v in obj.items()}
        if hasattr(obj, "to_dict"):
            return self._obj_to_plain(obj.to_dict())
        if hasattr(obj, "dict"):
            return self._obj_to_plain(obj.dict())
        if hasattr(obj, "__dict__"):
            return {k: self._obj_to_plain(v) for k, v in vars(obj).items()}
        return obj

    def _wrap_response(self, raw: Any) -> ChatCompletionFallback:
        plain = self._obj_to_plain(raw)
        return ChatCompletionFallback(plain)

    def _build_refusal_completion(self, model: str, cf_results: Dict[str, Any]) -> ChatCompletionFallback:
        fallback = {
            "id": None,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "finish_reason": "content_filter",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": "",
                    "refusal": True,
                    "role": "assistant",
                    "annotations": [],
                    "audio": None,
                    "function_call": None,
                    "tool_calls": None,
                },
            }],
            "usage": None,
            "prompt_filter_results": [{"prompt_index": 0, "content_filter_results": cf_results}],
        }
        return ChatCompletionFallback(fallback)

    async def _call_async(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: Optional[float],
        max_completion_tokens: Optional[int],
        stat_token_usage: bool = True,
        **kwargs
    ) -> ChatCompletionFallback:
        params = {"model": model, "messages": messages}
        if temperature is not None:
            params["temperature"] = temperature
        if max_completion_tokens is not None:
            params["max_completion_tokens"] = max_completion_tokens

        last_error = None
        retry_delay = self._retry_delay_seconds

        for attempt in range(self._max_retry_num):
            try:
                raw: ChatCompletion = await self._client.chat.completions.create(**params)
                if stat_token_usage:
                    self.token_usage.stat_token_usage(raw)
                if attempt > 0:
                    logger.info(f"Retry {attempt + 1}/{self._max_retry_num} succeeded")
                return self._wrap_response(raw)
            except (BadRequestError, InternalServerError) as e:
                cf_results = {}
                code = None
                try:
                    err = e.response.json().get("error", {})
                    code = err.get("code") or err.get("innererror", {}).get("code")
                    cf_results = err.get("innererror", {}).get("content_filter_result", {}) or {}
                except Exception:
                    pass
                logger.error(f"Error in GPT request: {e}, code: {code}")
                last_error = e
                if attempt < self._max_retry_num - 1:
                    logger.warning(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
            except Exception as e:
                logger.error(f"Error in GPT request: {e}")
                last_error = e
                if attempt < self._max_retry_num - 1:
                    logger.warning(f"Connection error on attempt {attempt + 1}/{self._max_retry_num}: {e}. Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    break

        logger.error(f"Max retries exceeded. Last error: {last_error}")
        return self._build_refusal_completion(model, {"error": f"Max retries exceeded: {str(last_error)}"})

    async def send_request(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4.1-mini",
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        json_mode: bool = False,
        stat_token_usage: bool = True,
        **kwargs
    ) -> ChatCompletionFallback:
        return await self._call_async(messages, model, temperature, max_completion_tokens, stat_token_usage, **kwargs)


    def request(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4.1-mini",
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        json_mode: bool = False,
        stat_token_usage: bool = True,
        **kwargs
    ) -> ChatCompletionFallback:
        return self._run_async(self._request_async(messages, model, temperature, max_completion_tokens, json_mode, stat_token_usage=stat_token_usage, **kwargs))


    def batch_requests(
        self,
        requests: List[Dict[str, Any]],
        progress_bar: bool = True,
        json_mode: bool = False,
        stat_token_usage: bool = True,
    ) -> List[ChatCompletionFallback]:
        return self._run_async(self._batch_requests_async(requests, progress_bar, json_mode, stat_token_usage))

    def _run_async(self, coro):
        try:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            return asyncio.run(coro)

    async def _request_async(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4.1-mini",
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        json_mode: bool = False,
        stat_token_usage: bool = True,
        **kwargs
    ) -> ChatCompletionFallback:
        return await self.send_request(messages, model, temperature, max_completion_tokens, json_mode=json_mode, stat_token_usage=stat_token_usage, **kwargs)

    async def _batch_requests_async(
        self,
        requests: List[Dict[str, Any]],
        progress_bar: bool = True,
        json_mode: bool = False,
        stat_token_usage: bool = True,
    ) -> List[ChatCompletionFallback]:
        tasks = []
        for req in requests:
            task = self.send_request(
                req["messages"],
                req.get("model", "gpt-4.1-mini"),
                req.get("temperature"),
                req.get("max_completion_tokens"),
                json_mode=json_mode,
                stat_token_usage=req.get("stat_token_usage", stat_token_usage),
                **{k: v for k, v in req.items() if k not in ("messages", "model", "temperature", "max_completion_tokens", "stat_token_usage")},
            )
            tasks.append(task)
        if progress_bar:
            return await tqdm.gather(*tasks, desc="Processing requests")
        else:
            return await asyncio.gather(*tasks)
