import random
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from hexagons.util import read_jsonl
from hexagons.board import COLORS_MAPPING


def calc_actions(
    state1: List[int],
    state2: List[int],
    is_one_based: bool,
    use_color_strings: bool
) -> List[Tuple]:
    actions = []
    for i, (c1, c2) in enumerate(zip(state1, state2)):
        if c1 == c2:
            continue
        row_num = i // 18 + (1 if is_one_based else 0)
        col_num = i % 18 + (1 if is_one_based else 0)
        c2 = c2 if not use_color_strings else COLORS_MAPPING[c2]
        actions.append((row_num, col_num, c2))
    return actions


def convert_to_t2t(
    procedure: List[Tuple],
    is_one_based: bool,
    use_color_strings: bool
):
    records = []
    input_template = "simplify instructions: {} simplified instructions: begin "
    prev_text = ""
    prev_state = None
    for step in procedure:
        instruction = step[1]
        state = step[2]
        if prev_state is not None:
            actions = calc_actions(prev_state, state, is_one_based, use_color_strings)
            source = input_template.format(instruction)
            target = " , ".join(["{} {} {}".format(*action) for action in actions])
            target += " end "
            records.append({
                "source": prev_text + source,
                "target": target,
                "instruction": instruction,
                "actions": actions,
                "prev_state": prev_state,
                "state": state
            })
            prev_text = prev_text + source + target
        prev_state = state
    return records


def read_records(
    file_name: str,
    is_one_based: bool = True,
    use_color_strings: bool = False
) -> List[Dict]:
    raw_records = read_jsonl(file_name)
    records = []
    for example in raw_records:
        records.extend(convert_to_t2t(example["drawing_procedure"], is_one_based, use_color_strings))
    return records


class T2TDataset(Dataset):
    def __init__(
        self,
        original_records: List[Dict],
        tokenizer: AutoTokenizer,
        max_source_tokens_count: int,
        max_target_tokens_count: int,
        sample_rate: float = 1.0,
        source_field: str = "source",
        target_field: str = "target"
    ):
        self.original_records = original_records
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.max_source_tokens_count = max_source_tokens_count
        self.max_target_tokens_count = max_target_tokens_count
        self.source_field = source_field
        self.target_field = target_field

        self.records = []
        for record in tqdm(original_records):
            if random.random() > self.sample_rate:
                continue
            tensors = self.convert_pair(
                source=record[source_field],
                target=record.get(target_field)
            )
            self.records.append(tensors)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def convert_pair(self, source, target):
        inputs = self.tokenizer(
            source,
            add_special_tokens=True,
            max_length=self.max_source_tokens_count,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        if target is not None:
            outputs = self.tokenizer(
                target,
                add_special_tokens=True,
                max_length=self.max_target_tokens_count,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            labels = outputs["input_ids"].squeeze(0)
            labels[outputs["attention_mask"].squeeze(0) == 0] = -100
            inputs["labels"] = labels
        return inputs
