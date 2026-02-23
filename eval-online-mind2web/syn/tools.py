import json
import base64
from PIL import Image
import io
import numpy as np
import hashlib
from datetime import datetime
from dataclasses import dataclass, fields, is_dataclass, MISSING
from enum import Enum
import os
from typing import Type, TypeVar, get_origin, get_args, Union
import types
import json_repair


def tools_serialize_dataclass(obj) -> dict:
    if isinstance(obj, Enum):
        return obj.value
    if not is_dataclass(obj):
        if isinstance(obj, set):
            return [tools_serialize_dataclass(item) for item in obj]
        if isinstance(obj, tuple):
            return [tools_serialize_dataclass(item) for item in obj]
        if isinstance(obj, list):
            return [tools_serialize_dataclass(item) for item in obj]
        if isinstance(obj, dict):
            return {key: tools_serialize_dataclass(value) for key, value in obj.items()}
        return obj
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    result = {}
    for field in fields(obj):
        value = getattr(obj, field.name)
        result[field.name] = tools_serialize_dataclass(value)
    return result


T = TypeVar('T')


def tools_deserialize_dataclass(data, target_type: Type[T]) -> T:
    if data is None:
        return None

    origin = get_origin(target_type)
    args = get_args(target_type)

    is_union = (origin is Union) or (getattr(types, "UnionType", None) is not None and origin is types.UnionType)
    if is_union:
        for arg_type in args:
            if arg_type is type(None):
                if data is None:
                    return None
                continue
            try:
                return tools_deserialize_dataclass(data, arg_type)
            except Exception:
                continue
        raise ValueError(f"Cannot deserialize {data!r} to any type in {target_type}")

    if origin is list:
        if not isinstance(data, list):
            raise ValueError(f"Expected list, got {type(data)}")
        item_type = args[0] if args else object
        return [tools_deserialize_dataclass(item, item_type) for item in data]

    if origin is tuple:
        if not isinstance(data, list):
            raise ValueError(f"Expected list for tuple, got {type(data)}")
        if not args:
            return tuple(data)
        result_items = []
        for i, item in enumerate(data):
            arg_type = args[i] if i < len(args) else (args[-1] if args else object)
            result_items.append(tools_deserialize_dataclass(item, arg_type))
        return tuple(result_items)

    if origin is set:
        if not isinstance(data, list):
            raise ValueError(f"Expected list for set, got {type(data)}")
        item_type = args[0] if args else object
        return {tools_deserialize_dataclass(item, item_type) for item in data}

    if origin is dict:
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data)}")
        key_type = args[0] if args else str
        value_type = args[1] if len(args) > 1 else object
        return {
            tools_deserialize_dataclass(k, key_type): tools_deserialize_dataclass(v, value_type)
            for k, v in data.items()
        }

    if isinstance(target_type, type) and issubclass(target_type, Enum):
        return target_type(data)

    if isinstance(target_type, type) and is_dataclass(target_type):
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict for dataclass {target_type}, got {type(data)}")
        if not os.environ.get('DISABLE_FROM_DICT', False) and hasattr(target_type, 'from_dict'):
            return target_type.from_dict(data)
        field_values = {}
        for field in fields(target_type):
            if field.name in data:
                field_values[field.name] = tools_deserialize_dataclass(data[field.name], field.type)
            elif field.default is not MISSING:
                field_values[field.name] = field.default
            elif field.default_factory is not MISSING:
                field_values[field.name] = field.default_factory()
            else:
                raise ValueError(f"Missing required field {field.name} for {target_type}")
        obj = target_type.__new__(target_type)
        for field_name, field_value in field_values.items():
            setattr(obj, field_name, field_value)
        return obj

    if target_type in (int, float, str, bool):
        return target_type(data)

    return data


def tools_jsonl_load(path: str) -> list[dict]:
    with open(path, 'r') as f:
        return [json.loads(line) for line in f.readlines()]


def tools_jsonl_save(data: list[dict], path: str, append: bool = False):
    mode = 'a' if append else 'w'
    with open(path, mode, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def tools_load_png_rgba(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
    return np.array(img)


def tools_ndarray_image_save(arr: np.ndarray, path: str):
    if arr.shape[2] == 4:
        background = np.ones_like(arr[:, :, :3], dtype=np.uint8) * 255
        alpha = arr[:, :, 3:] / 255.0
        rgb = (arr[:, :, :3] * alpha + background * (1 - alpha)).astype(np.uint8)
    else:
        rgb = arr
    img = Image.fromarray(rgb)
    img.save(path, format="JPEG")


def tools_ndarray_to_base64_image(arr: np.ndarray) -> str:
    if arr.shape[2] == 4:
        background = np.ones_like(arr[:, :, :3], dtype=np.uint8) * 255
        alpha = arr[:, :, 3:] / 255.0
        rgb = (arr[:, :, :3] * alpha + background * (1 - alpha)).astype(np.uint8)
    else:
        rgb = arr
    img = Image.fromarray(rgb)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def tools_ndarray_to_base64_image_raw(arr: np.ndarray, format: str = "JPEG") -> tuple[str, tuple[int, int]]:
    if len(arr.shape) == 3 and arr.shape[2] == 4:
        background = np.ones_like(arr[:, :, :3], dtype=np.uint8) * 255
        alpha = arr[:, :, 3:] / 255.0
        rgb = (arr[:, :, :3] * alpha + background * (1 - alpha)).astype(np.uint8)
    elif len(arr.shape) == 3 and arr.shape[2] == 3:
        rgb = arr
    else:
        rgb = arr
    img = Image.fromarray(rgb)
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode("ascii")
    return b64, (img.width, img.height)


def tools_get_time() -> str:
    return datetime.now().strftime("%y-%m-%d-%H_%M_%S")


def tools_elapsed_time_print(start_time_str: str) -> str:
    start_time = datetime.strptime(start_time_str, "%y-%m-%d-%H_%M_%S")
    now = datetime.now()
    elapsed = now - start_time
    days = elapsed.days
    total_seconds = elapsed.seconds
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"Elapsed time: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds"


def tools_hash(hash_str: str) -> int:
    digest = hashlib.sha256(hash_str.encode("utf-8")).hexdigest()
    return int(digest, 16)


def tools_robust_json_loads(s: str) -> dict:
    decoded = json_repair.loads(s)
    if len(decoded) > 0:
        return decoded
    raise ValueError(f"Failed to decode JSON string='{s}' even after repair.")
