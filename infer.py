import os
import json
import torch
import pandas as pd
from copy import deepcopy
from itertools import groupby
from transformers import HfArgumentParser, Cache
from utils import load_model_and_tokenizer
from dataprocessor import SamplePreprocessor, CHADataCollator
from dataclasses import dataclass, field
from typing import Optional, Union, List
from tqdm import tqdm
import time

@dataclass
class InferArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    param_dir: str = field(
        default=None, metadata={"help": "The directory to load the parameters from"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer to use."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default="eager",
        metadata={
            "help": ("The attention implementation to use in the model."),
            "choices": ["eager", "sdpa", "flash_attention_2"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to folder with train.json and val.json"},
    )
    save_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the processed dataset"},
    )
    beacon_size: int = field(
        default=1, metadata={"help": "Beacon size"}
    )

@torch.no_grad()
def prefill(
    model,
    input_ids: torch.LongTensor,
    segment_ids: torch.LongTensor,
    is_beacon: Optional[torch.BoolTensor] = None,
):
    # segment_ids is a list of segment ids for each token in the input_ids
    # e.g. [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 0, 0, 0, 0]
    # We need to extract each segment and iteratively construct the inputs for generation

    # First, split segment_ids by the same value, e.g. we get a 2D list [[0, 0, 0], [1, 1, 1], [2, 2], [3, 3, 3, 3], [4, 4, 4], [0, 0, 0, 0]]
    
    segment_ids = segment_ids.squeeze().tolist()
    # print(segment_ids)
    segments = [list(group) for key, group in groupby(segment_ids)]
    segments_lens = [len(segment) for segment in segments]
    segments_start_end = [(0, segments_lens[0])]
    for i in range(1, len(segments_lens)):
        segments_start_end.append((segments_start_end[-1][1], segments_start_end[-1][1] + segments_lens[i]))

    attention_mask = torch.ones((input_ids.shape[0], input_ids.shape[1]), dtype=torch.long, device=input_ids.device)
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    # 1. I want to extract the cur_input_ids in the same segment
    past_key_values = None # FIXME
    for s, e in segments_start_end:
        # print(s, e)
        cur_input_ids = input_ids[:, s:e]
        cur_position_ids = position_ids[:, s:e]
        cur_is_beacon = is_beacon[:, s:e]
        # print(cur_is_beacon.sum())
        outputs = model(
            input_ids=cur_input_ids,
            position_ids=cur_position_ids,
            past_key_values = past_key_values,
            use_cache=True,
        )
        # true_indices = torch.nonzero(is_beacon, as_tuple=True)
        # is_table = len(true_indices[1]) > 0
        is_table = cur_is_beacon.any().item()
        if is_table:
            # retain_mask = is_beacon.clone()
            # first_true_index = true_indices[1][0]
            # retain_mask[:, first_true_index:] = True
            # cat the past_key_values 
            retain_mask = cur_is_beacon.squeeze()
            if past_key_values is None:
                past_key_values = [
                    (
                        key[:, :, retain_mask, :], value[:, :, retain_mask, :],
                    )
                    for key, value in outputs.past_key_values
                ]
            else:
                past_key_values = [
                    (
                        torch.cat([key1, key2[:, :, -len(retain_mask):, :][:, :, retain_mask, :]], dim=2),
                        torch.cat([value1, value2[:, :, -len(retain_mask):, :][:, :, retain_mask, :]], dim=2),
                    )
                    for (key1, value1), (key2, value2) in zip(
                        past_key_values, outputs.past_key_values
                    )
                ]
        else:
            past_key_values = outputs.past_key_values
    logits = outputs.logits[:, -1, :]  # (1, vocab_size)
    next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (1, 1)

    # print("Previous length: ", position_ids.shape[1])
    # print("Now length: ", past_key_values[0][0].shape[2])
    old_length = position_ids.shape[1]
    new_length = past_key_values[0][0].shape[2]
    input_ids = next_token
    attention_mask = torch.ones((input_ids.shape[0], new_length + input_ids.shape[1]), dtype=torch.long, device=input_ids.device)
    position_ids = torch.LongTensor([[old_length]]).to(input_ids.device)
    return input_ids, attention_mask, position_ids, past_key_values

@torch.no_grad()
def decode(
    model,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
    position_ids: torch.LongTensor,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    max_new_tokens: int = 8000,
    eos_token_id: Optional[int] = None,
) -> torch.LongTensor:
    model.eval()
    device = input_ids.device
    assert input_ids.size(0) == 1, "Only batch size = 1 is supported."

    eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id
    if not isinstance(eos_token_id, list):
        eos_token_id = [eos_token_id]

    for step in range(max_new_tokens):
        # 只取当前 token 作为输入
        cur_input_ids = input_ids[:, -1:]
        cur_position_ids = position_ids[:, -1:]

        outputs = model(
            input_ids=cur_input_ids,
            attention_mask=attention_mask,
            position_ids=cur_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

        logits = outputs.logits[:, -1, :]  # (1, vocab_size)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (1, 1)

        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if next_token.item() in eos_token_id:
            break

        # 更新 position_ids、attention_mask、past_key_values
        next_position = position_ids[:, -1:] + 1
        position_ids = torch.cat([position_ids, next_position], dim=1)

        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)],
            dim=1,
        )
        past_key_values = outputs.past_key_values

    return input_ids

if __name__ == "__main__":
    parser = HfArgumentParser(InferArguments)
    (args,) = parser.parse_args_into_dataclasses()

    if "llama" in args.model_name_or_path.lower():
        if "3.1" in args.model_name_or_path.lower():
            model_name = "llama3.1"
        else:
            model_name = "llama3"
    elif "qwen" in args.model_name_or_path.lower():
        model_name = "qwen"
    elif "mistral" in args.model_name_or_path.lower():
        model_name = "mistral"
    elif "deepseek" in args.model_name_or_path.lower():
        model_name = "deepseek"
    else:
        raise ValueError("Unsupported model name. Please use a model from Llama, Qwen, or Mistral.")

    print("Model name:", model_name)

    model, tokenizer = load_model_and_tokenizer(model_args=args, param_dir=args.param_dir)
    model_name = args.param_dir.split('/')[-1] if args.param_dir else "unknown"
    model = model.cuda()
    model.eval()

    preprocessor = SamplePreprocessor(tokenizer=tokenizer, beacon_size=args.beacon_size)
    data_collator = CHADataCollator()

    fnames = [x for x in os.listdir(args.dataset_path) if x.endswith('.jsonl')]
    for filename in fnames:
        print(f"Processing {filename}")
        if "DP" in filename:
            instruction_type = "DP"
        elif "PoT" in filename:
            instruction_type = "PoT"
        elif "SCoT" in filename:
            instruction_type = "SCoT"
        elif "TCoT" in filename:
            instruction_type = "TCoT"
        else:
            raise ValueError("Not support")

        file_path = os.path.join(args.dataset_path, filename)
        lines = [json.loads(x) for x in open(file_path, encoding='utf-8').readlines() if x.strip()]

        # 保存路径
        save_path = os.path.join(args.save_path, args.param_dir.split('/')[-1] + '_' + filename.split('.')[0] + '.jsonl')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # resume: 加载已完成的 prediction（如果有）
        if os.path.exists(save_path):
            with open(save_path, 'r', encoding='utf-8') as f:
                saved = {json.loads(l)['id']: json.loads(l) for l in f if '"prediction"' in l}
        else:
            saved = {}

        # 初始化计时变量
        total_prefill_time = 0.0
        total_decode_time = 0.0
        processed_count = 0
        
        with open(save_path, 'w') as fout:
            for idx, line in tqdm(enumerate(lines), total=len(lines)):
                torch.cuda.empty_cache()
                sample_id = line['id']
                if sample_id in saved:
                    # 如果已经处理过，从保存的数据中恢复时间统计
                    if 'prefill_time' in saved[sample_id] and 'decode_time' in saved[sample_id]:
                        total_prefill_time += saved[sample_id]['prefill_time']
                        total_decode_time += saved[sample_id]['decode_time']
                        processed_count += 1
                    fout.write(json.dumps(saved[sample_id]) + '\n')
                    continue

                sample = deepcopy(line)
                table_dict = sample['table']
                df = pd.DataFrame(table_dict["data"], columns=table_dict["columns"])
                sample['df'] = df
                sample['instruction_type'] = instruction_type
                processed_sample = preprocessor(sample)
                # sample = data_collator([processed_sample])
                sample = processed_sample

                input_ids = torch.LongTensor(sample['input_ids']).unsqueeze(0).cuda()
                segment_ids = torch.LongTensor(sample['segment_ids']).unsqueeze(0).cuda()  # (1, L)
                is_beacon = torch.tensor(sample['is_beacon'], dtype=torch.bool).unsqueeze(0).cuda()

                # input_ids = torch.LongTensor(sample['input_ids']).unsqueeze(0)
                # segment_ids = torch.LongTensor(sample['segment_ids']).unsqueeze(0)
                # is_beacon = torch.tensor(sample['is_beacon'], dtype=torch.bool).unsqueeze(0)
                # input_ids = sample['input_ids']
                # attention_mask = sample['attention_mask']
                # position_ids = sample['position_ids']
                # question_ids = sample['question_ids']
                # is_beacon = sample['is_beacon']

                # 记录prefill时间
                prefill_start = time.time()
                input_ids, attention_mask, position_ids, past_key_values = prefill(
                    model = model,
                    input_ids=input_ids,
                    segment_ids=segment_ids,
                    is_beacon=is_beacon
                )
                prefill_time = time.time() - prefill_start
                total_prefill_time += prefill_time

                inputs = {
                    "model": model,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "past_key_values": past_key_values,
                }

                # 记录decode时间
                decode_start = time.time()
                output_ids = decode(
                    **inputs,
                    max_new_tokens=8000,
                )
                decode_time = time.time() - decode_start
                total_decode_time += decode_time
                processed_count += 1

                # generated_texts = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
                generated_texts = tokenizer.decode(output_ids[0], skip_special_tokens=True)

                print(f"{idx}: generated: {generated_texts}, answer: {line['answer']}")

                line['prediction'] = generated_texts
                line['model_name'] = model_name
                # 保存时间信息用于resume
                line['prefill_time'] = prefill_time
                line['decode_time'] = decode_time

                fout.write(json.dumps(line) + '\n')

        # 计算并输出平均时间
        if processed_count > 0:
            avg_prefill = total_prefill_time / processed_count
            avg_decode = total_decode_time / processed_count
            print(f"File {filename}: Average prefill time: {avg_prefill:.4f}s, Average decode time: {avg_decode:.4f}s")
            print(f"File {filename}: Total processed samples: {processed_count}")
        else:
            print(f"File {filename}: No new samples processed")