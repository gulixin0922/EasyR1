# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
import itertools
import re
import time
import torch
import math
import transformers
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2_5_VLProcessor, Qwen2VLImageProcessor
from qwen_vl_utils import process_vision_info
from copy import deepcopy
from typing import Dict


SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."""


ds_collections = {
    'CMMU-Base':{
        'root': '/mnt/petrelfs/share_data/gulixin/chemvlm/cmmu',
        'question': '/mnt/petrelfs/share_data/gulixin/chemvlm/cmmu/chemistry_conv.jsonl',
        'prompt': "回答上面的考题，请先输出解析再输出最终答案。",
        # 'prompt': SYSTEM_PROMPT,
        'max_new_tokens': 4000,
        'min_new_tokens': 1,
    },
    'CMMU-Thinking':{
        'root': '/mnt/petrelfs/share_data/gulixin/chemvlm/cmmu',
        'question': '/mnt/petrelfs/share_data/gulixin/chemvlm/cmmu/chemistry_conv.jsonl',
        'prompt': SYSTEM_PROMPT,
        'max_new_tokens': 4000,
        'min_new_tokens': 1,
    },
    'mmcr_post-Base':{
        'root': '/mnt/petrelfs/share_data/gulixin/datasets/chemistry_postgraduate_examination_qa',
        'question': '/mnt/petrelfs/share_data/gulixin/chemvlm/mmcr_post/test_set_all_filtered.jsonl',
        'prompt': "回答上面的考题，请先输出解析再输出最终答案。",
        'max_new_tokens': 4000,
        'min_new_tokens': 1,
    },
    'mmcr_post-Thinking':{
        'root': '/mnt/petrelfs/share_data/gulixin/datasets/chemistry_postgraduate_examination_qa',
        'question': '/mnt/petrelfs/share_data/gulixin/chemvlm/mmcr_post/test_set_all_filtered.jsonl',
        'prompt': SYSTEM_PROMPT,
        'max_new_tokens': 4000,
        'min_new_tokens': 1,
    },
}


class chemDataset(torch.utils.data.Dataset):
    def __init__(self, root, data, prompt):
        self.root = root
        self.data = open(data).readlines()
        self.prompt = prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = json.loads(self.data[idx].strip())
        question_id = data_item['id']
        annotation = data_item['conversations'][1]["value"] #TODO
        if 'image' in data_item:
            image_path_list = data_item['image']
        elif 'images' in data_item:
            image_path_list = data_item['images']
        else:
            image_path_list = []
        
        if isinstance(image_path_list, str):
            image_path_list = [image_path_list]
        
        image_path_list = [os.path.join(self.root, image_path) for image_path in image_path_list]
        question  = data_item['conversations'][0]['value']
        return {
            'question_id': question_id,
            'question': question,
            'annotation': annotation,
            'image_path_list': image_path_list,
            'system_prompt': self.prompt,
        }
    

def collate_fn(batches):
    question_ids = [_['question_id'] for _ in batches]
    questions = [_['question'] for _ in batches]
    annotations = [_['annotation'] for _ in batches]
    image_path_list= [_['image_path_list'] for _ in batches]
    system_prompts = [_['system_prompt'] for _ in batches]
    return question_ids, questions, annotations, image_path_list, system_prompts


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def process_image(image, min_pixels, max_pixels):
    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def evaluate_chat_model():
    random.seed(args.seed)

    for ds_name in args.datasets:
        dataset = chemDataset(
            root=ds_collections[ds_name]['root'],
            data=ds_collections[ds_name]['question'],
            prompt=ds_collections[ds_name]['prompt'],
        )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        outputs = []
        for _, (question_ids, questions, annotations, image_path_list, system_prompts) in tqdm(enumerate(dataloader)):
            question = questions[0]
            system_prompt = system_prompts[0]
            if 'Thinking' in ds_name:
                messages = [{"role": "user", "content": question}]
                messages.insert(0, {"role": "system", "content": system_prompt})
            else:
                messages = [{"role": "user", "content": question + "\n" + system_prompt}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            image_inputs = [process_image(Image.open(image_path), args.min_pixels, args.max_pixels) for image_path in image_path_list[0]]
            if len(image_inputs) > 0:
                inputs = processor(images=image_inputs, text=prompt, return_tensors="pt")
            else:
                inputs = processor(images=None, text=prompt, return_tensors="pt")
            # inputs = inputs.to(f'cuda:{int(os.getenv("LOCAL_RANK", 0))}')
            inputs = inputs.to("cuda")
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                min_new_tokens=ds_collections[ds_name]['min_new_tokens'],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            outputs.append({
                'id': question_ids[0],
                'text': output_text[0],
                'annotation': annotations[0],
                'metadata': {}
            })
            print('prompt:', prompt)
            print('annotation:', annotations[0])
            print('output_text:', output_text[0])

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')

            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.jsonl'
            results_file = os.path.join(args.out_dir, results_file)
            writer = open(results_file, 'w')
            for item in merged_outputs:
                writer.write(json.dumps(item, ensure_ascii=False) + '\n')
            writer.close()
            print('Results saved to {}'.format(results_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='CMMU')#qinghao_test，mm_gaokao_test_past,Chembench_mol2cap,Chembench_yield,Chembench_property,Chembench_name_conv,Chembench_retro,Chembench_temperature,Chembench_solvent,molecule,smiles_ocr,chirality_yon,chirality_num,SciQA,CMMU,orderly_product,chem_free_response,scibench_matter,scibench_chemmc,scibench_quan,scibench_atkins,chem_multiple_choice,chem_free_response,mmsci
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--min-pixels', type=int, default=262144)
    parser.add_argument('--max-pixels', type=int, default=4194304)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16, device_map=f"cuda:{int(os.getenv('LOCAL_RANK', 0))}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16).cuda().eval()
    processor = AutoProcessor.from_pretrained(args.checkpoint)

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f'[test] total_params: {total_params}B')

    evaluate_chat_model()


