import copy
import json
import math
import os
import random
import re
from typing import Dict

import torch
import transformers
import yaml
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset

from aguvis.constants import (
    IGNORE_INDEX,
    additional_special_tokens,
    assistant_template,
    chat_template,
    grounding_system_message,
)
from aguvis.trainer import rank0_print


class LazySupervisedDataset(Dataset):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        processor: transformers.ProcessorMixin,
        data_path: str,
        data_args,
        use_action_weight: bool = True,
        action_weight: float = 2.0,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.processor = processor
        self.use_action_weight = use_action_weight
        self.action_weight = action_weight
        self.list_data_dict = []
        self.list_image_path = []
        # print action weight for debugging
        print('#############################################')
        print(f'Use action weight: {self.use_action_weight}, action weight: {self.action_weight}')
        print('#############################################')

        # Handle multiple JSON files specified in the data_path
        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            rank0_print(f"Loading {file_names} from {base_path}")
            data_args.dataset_paths = []
            for file_name in file_names:
                data_args.dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                rank0_print(f"Loading {full_path}")
                with open(full_path) as file:
                    cur_data_dict = json.load(file)
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                    self.list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".yaml"):
            with open(data_path) as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999
                data_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
                for dataset in datasets:
                    json_path = os.path.expandvars(dataset.get("json_path"))
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    images_folder = dataset.get("images_folder")
                    sampling_number = None

                    rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path) as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        # NOTE: we only use json_path with .json now
                        # Handle the images_folder in yaml
                        with open(json_path) as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]

                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
                    self.list_image_path.extend([images_folder] * len(cur_data_dict))
        else:
            data_args.dataset_paths = [data_path]
            rank0_print(f"Loading {data_path}")
            with open(data_path) as file:
                cur_data_dict = json.load(file)
                rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                self.list_data_dict.extend(cur_data_dict)
                self.list_image_path.extend([""] * len(cur_data_dict))  # NOTE: the image subfolder is empty...

        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = (
                1200 * len(sample["image"]) if isinstance(sample["image"], list) else 1200 if "image" in sample else 0
            )
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            assert cur_len > 0, f"Conversation length is 0 for {sample}"

            img_tokens = (
                1200 * len(sample["image"]) if isinstance(sample["image"], list) else 1200 if "image" in sample else 0
            )

            if "image" in sample or "video" in sample or self.data_args.early_mix_text:
                length_list.append(cur_len + img_tokens)
            else:
                length_list.append(-cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            sample = self._get_item(i)
        except Exception as e:
            print(f"Failed to fetch sample {i}. Exception:", e)
            new_index = random.randint(0, len(self.list_data_dict) - 1)
            return self.__getitem__(new_index)
        return sample

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        image_path = os.path.join(self.data_args.image_folder, self.list_image_path[i])
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"

        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            if type(image_file) is list:
                image = [os.path.join(image_path, image_file) for image_file in image_file]
            else:
                image = [os.path.join(image_path, image_file)]

            sources = copy.deepcopy([e["conversations"] for e in sources])
        elif "video" in sources[0]:
            raise NotImplementedError("Video is not supported for Qwen2VL")
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = self.preprocess_qwen2vl(sources, self.tokenizer, self.processor, image)
        if isinstance(i, int):
            new_data_dict = {
                "input_ids": data_dict["input_ids"][0],
                "labels": data_dict["labels"][0],
            }
            if "pixel_values" in data_dict:
                new_data_dict["pixel_values"] = data_dict["pixel_values"]
                new_data_dict["image_grid_thw"] = data_dict["image_grid_thw"]

            if self.use_action_weight:
                new_data_dict["action_weights"] = data_dict["action_weights"][0]
            data_dict = new_data_dict

        data_dict["id"] = self.list_data_dict[i].get("id", i)
        return data_dict




    def preprocess_qwen2vl(
        self,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        processor: transformers.ProcessorMixin,
        image: list,
        system_message: str = grounding_system_message,
        agent_mode: bool = True,
        chat_template: str = chat_template,
        assistant_template: str = assistant_template,
    ) -> Dict:
        roles = {"human": "user", "gpt": "assistant", "system": "system"}
        # assistant_template = assistant_template if agent_mode else chat_template
        processor.tokenizer = tokenizer

        image_index = 0
        # Apply prompt templates
        input_ids, targets = [], []
        pixel_values, image_grid_thw = None, None
        for source in sources:
            if roles[source[0]["from"]] == "system":
                system_message = source[0]["value"]
                source = source[1:]

            input_id, target = [], []
            convs = []

            # New version, use apply chat template
            # Build system message for each sentence
            input_id += tokenizer.apply_chat_template(
                conversation=[{"role": "system", "content": [{"type": "text", "text": system_message}]}],
                chat_template=chat_template,
            )
            convs.append({"role": "system", "content": [{"type": "text", "text": system_message}]})
            target += [IGNORE_INDEX] * len(input_id)

            if 'recipient' in source[-1] and source[-1]['recipient'] == 'os':
                # move the action to the second last conversation
                if 'value' in source[-1]:
                    source[-2]['value'] += '\nExcutable Action: ' + source[-1]['value']
                elif 'content' in source[-1]:
                    source[-2]['content'] += '\nExcutable Action: ' + source[-1]['content']
                else:
                    raise ValueError("The last conversation should have 'value' or 'content' field")
                # remove the last conversation
                source = source[:-1]

            for conv in source:
                # Make sure llava data can load
                try:
                    role = conv["role"]
                    content = conv["content"]
                except Exception:
                    role = conv["from"]
                    content = conv["value"]

                role = roles.get(role, role)

                # Count the number of <image> tokens in the content
                image_count = content.count("<image>")
                # If there are images, add them to the content
                if image_count > 0:
                    assert role == "user", "Images are only supported for user messages"
                    image_placeholders = []
                    for _ in range(image_count):
                        image_placeholders.append({"type": "image", "image": image[image_index]})
                        image_index += 1

                    content = content.replace("<image>", "")
                    conv = [{"role": role, "content": image_placeholders + [{"type": "text", "text": content}]}]
                    convs += conv
                    ########################################################
                    patch_size = 16 if "Qwen3-VL" in getattr(tokenizer, "name_or_path", "") else 14
                    image_inputs, _ = process_vision_info(conv, image_patch_size=patch_size)
                    # image_inputs, _ = process_vision_info(conv)
                    templated_conv = tokenizer.apply_chat_template(
                        conversation=conv, tokenize=False, chat_template=chat_template,
                    )
                    inputs = processor(text=[templated_conv], images=image_inputs, return_tensors="pt")
                    if pixel_values is None and image_grid_thw is None:
                        pixel_values = inputs["pixel_values"]
                        image_grid_thw = inputs["image_grid_thw"]
                    else:
                        pixel_values = torch.concat([pixel_values, inputs["pixel_values"]], dim=0)
                        image_grid_thw = torch.concat([image_grid_thw, inputs["image_grid_thw"]], dim=0)

                else:
                    if role in ["user", "system"]:
                        conv = [{"role": role, "content": [{"type": "text", "text": content}]}]
                    else:  # assistant
                        conv = [
                            {
                                "role": role,
                                "content": [{"type": "text", "text": content}],
                            }
                        ]
                    convs += conv
                    templated_conv = tokenizer.apply_chat_template(
                        conversation=conv,
                        tokenize=False,
                        chat_template=chat_template,
                    )
                    inputs = processor(text=[templated_conv], return_tensors="pt")

                encode_id = inputs.input_ids[0].tolist()

                input_id += encode_id
                if role in ["user", "system"]:
                    target += [IGNORE_INDEX] * len(encode_id)
                else:
                    target += encode_id

            assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
            # for idx, encode_id in enumerate(input_id):
            #     if encode_id in unmask_tokens_idx:
            #         target[idx] = encode_id
            input_ids.append(input_id)
            targets.append(target)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)

        # calculate weight for action tokens if needed
        if self.use_action_weight:
            # 1. 先拿到 "<think>" / "</think>" / "<answer>" / "</answer>" 的「多 token pattern」
            #    注意：这里不再假设它们是单个 token，而是允许被 tokenizer 切成多片
            def get_pattern_ids(text: str):
                # 不加 special tokens，只对这几个字符串本身做分词
                return tokenizer(text, add_special_tokens=False).input_ids

            answer_start_pat = get_pattern_ids("<answer>")[:-1]
            answer_end_pat   = get_pattern_ids("</answer>")[:-1]
            grounding_start_pat = get_pattern_ids('"point_2d":')[1:]
            grounding_end_pat   = get_pattern_ids(']\n')

            # 2. 辅助函数：在 seq 里找到 pat 这个子序列所有起始位置
            def find_all_pattern_positions(seq, pat, start_pos=0):
                positions = []
                n, m = len(seq), len(pat)
                if m == 0 or n < m:
                    return positions
                for i in range(start_pos, n - m + 1):
                    if seq[i : i + m] == pat:
                        positions.append(i)
                return positions

            # 初始化所有 token 的权重为 1
            action_weights = torch.ones_like(targets, dtype=torch.float)

            bs, seqlen = input_ids.size()
            input_ids_list = input_ids.tolist()  # 转成 list，方便 Python 层匹配子序列

            for b in range(bs):
                seq = input_ids_list[b]

                # 找到所有 <answer> / </answer> 的位置
                answer_starts = find_all_pattern_positions(seq, answer_start_pat)
                answer_ends   = find_all_pattern_positions(seq, answer_end_pat, start_pos=answer_starts[0] if len(answer_starts) else 0)
                # print(answer_start_pat, answer_end_pat)
                # print(answer_starts, answer_ends)
                # 找到所有 "point_2d": / ] 的位置
                grounding_starts = find_all_pattern_positions(seq, grounding_start_pat)
                # grounding_ends   = find_all_pattern_positions(seq, grounding_end_pat, start_pos=grounding_starts[0] if len(grounding_starts) else 0)

                # if random.random() < 0.0:
                #     print(len(answer_starts), len(answer_ends), len(grounding_starts))
                # 只匹配第一对
                if len(answer_starts) > 0 and len(answer_ends) > 0:
                    start = answer_starts[0] + len(answer_start_pat) 
                    end   = answer_ends[0] 
                    action_weights[b, start:end] *= self.action_weight
                if len(grounding_starts) > 0:
                    start = grounding_starts[0] + len(grounding_start_pat)
                    action_weights[b, start:] *= self.action_weight * 2

        data_dict = {
            "input_ids": input_ids,  # tensor(bs x seq_len)
            "labels": targets,       # tensor(bs x seq_len)
        }
        if self.use_action_weight:
            data_dict["action_weights"] = action_weights

        if pixel_values is not None:
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_grid_thw

        return data_dict


        