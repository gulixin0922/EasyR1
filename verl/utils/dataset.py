# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF

try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config
except ImportError as E:
    print('petrel_client is not installed. If you read data locally instead of from ceph, ignore it.')
import sys
import json
import random
import copy
client = Client('~/petreloss.conf')


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


class ImageProcessMixin:
    max_pixels: int
    min_pixels: int

    def process_image(self, image: Union[Dict[str, Any], ImageObject]) -> ImageObject:
        if isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))

        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image


def pil_loader(img_str):
    buff = BytesIO(img_str)
    img = Image.open(buff)
    return img.convert('RGB')


class RLHFDataset(Dataset, ImageProcessMixin):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        system_prompt: str = None,
        max_pixels: int = None,
        min_pixels: int = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isfile(data_path) and data_path.endswith('.jsonl'):
            with open(data_path, 'r') as f:
                self.dataset = f.readlines()
                self.dataset = [json.loads(line) for line in self.dataset]
        elif os.path.isdir(data_path):
            self.dataset = load_dataset("parquet", data_dir=data_path, split="train")
        elif os.path.isfile(data_path):
            self.dataset = load_dataset("parquet", data_files=data_path, split="train")
        else:  # remote dataset
            self.dataset = load_dataset(data_path, split=data_split)

    def __len__(self):
        return len(self.dataset)

    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        if 's3://' in image_path:
            img_value_str = client.get(image_path)
            img = pil_loader(img_value_str)
            return img
        return Image.open(image_path).convert('RGB')

    def __getitem__(self, index):
        while True:
            row_dict: dict = copy.deepcopy(self.dataset[index])
            messages = [{"role": "user", "content": row_dict[self.prompt_key]}]
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})

            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            if self.image_key in row_dict:
                if isinstance(row_dict[self.image_key][0], str):
                    for img_id, image_path in enumerate(row_dict[self.image_key]):
                        row_dict[self.image_key][img_id] = self.load_image(image_path)
                    
                prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
                row_dict["multi_modal_data"] = {
                    "image": [
                        process_image(image, self.max_pixels, self.min_pixels) for image in row_dict.pop(self.image_key)
                    ]
                }
                model_inputs = self.processor(row_dict["multi_modal_data"]["image"], prompt, return_tensors="pt")
                input_ids = model_inputs.pop("input_ids")[0]
                attention_mask = model_inputs.pop("attention_mask")[0]
                row_dict["multi_modal_inputs"] = dict(model_inputs)
                position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids,
                    image_grid_thw=model_inputs["image_grid_thw"],
                    attention_mask=attention_mask,
                )  # (3, seq_length)
            else:
                model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
                input_ids = model_inputs.pop("input_ids")[0]
                attention_mask = model_inputs.pop("attention_mask")[0]
                position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)
            
            if len(input_ids) > self.max_prompt_length:
                print("--------------------------------")
                print("prompt:", prompt)
                print("--------------------------------")
                print("input_ids length:", len(input_ids))
                index = random.randint(0, len(self.dataset) - 1)
                continue
            else:
                input_ids, attention_mask, position_ids = VF.postprocess_data(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    max_length=self.max_prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,
                    truncation=self.truncation,
                )
                row_dict["input_ids"] = input_ids
                row_dict["attention_mask"] = attention_mask
                row_dict["position_ids"] = position_ids
                row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
                row_dict["ground_truth"] = row_dict.pop(self.answer_key)
                break
        return row_dict
