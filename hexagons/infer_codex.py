import argparse
import os
import json
import traceback
import sys
from typing import List, Tuple
from time import sleep

from tqdm import tqdm

from hexagons.codex import codex_predict
from hexagons.metrics import Metrics
from hexagons.dataset import calc_actions, apply_actions
from hexagons.board import COLORS_MAPPING
from hexagons.util import read_jsonl


parser = argparse.ArgumentParser()
parser.add_argument("input_file", type=str)
parser.add_argument("output_file", type=str)
parser.add_argument("--delay-sec", type=int, default=20)
parser.add_argument("--prompt-file", type=str, required=True)
args = parser.parse_args()

with open(args.prompt_file) as r:
    original_prompt = r.read().strip()

metrics = Metrics()
raw_records = read_jsonl(args.input_file)
output_records = []
for example in tqdm(raw_records):
    index = example["index"]
    procedure = example["drawing_procedure"]
    prev_state = None
    cur_state = None
    prompt = original_prompt
    for step in procedure:
        instruction, state = step[1], step[2]
        output_record = {
            "instruction": instruction,
            "prompt": prompt,
            "state": state
        }
        instruction = " ".join(instruction.split())
        if prev_state is None:
            prev_state = state
            cur_state = state
            continue
        true_actions = calc_actions(prev_state, state, True, True)
        output_record["true_actions"] = true_actions

        pred_actions = []
        new_prompt = prompt + " " + instruction + "\n"
        program = codex_predict(new_prompt)
        try:
            output_record["program"] = program
            exec(program)
            pred_actions = grid.get_actions()
            program += "\n# Cleaning\ngrid.clean_actions()\n"
            new_prompt = program + "\n#"
        except Exception as e:
            output_record["error"] = traceback.format_exc()
            print(output_record["error"])
            new_prompt = prompt

        prompt = new_prompt
        output_record["pred_actions"] = pred_actions
        cur_state = apply_actions(cur_state, pred_actions)
        metrics.add(true_actions, pred_actions, state, cur_state)
        sleep(args.delay_sec)
        prev_state = state
        output_records.append(output_record)
        metrics.print_all()

metrics.print_all()

with open(args.output_file, "w") as w:
    for r in output_records:
        w.write(json.dumps(r, ensure_ascii=False).strip() + "\n")

