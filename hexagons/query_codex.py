import argparse
import os
import json
import traceback
import sys
from typing import List, Tuple
from time import sleep

import openai

from hexagons.dataset import calc_actions
from hexagons.board import COLORS_MAPPING
from hexagons.util import read_jsonl


def codex_predict(prompt):
    while True:
        try:
            response = openai.Completion.create(
                model="code-davinci-002",
                prompt=prompt,
                temperature=0.1,
                max_tokens=256,
                top_p=1,
                best_of=5,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["#"]
            )
            break
        except openai.error.RateLimitError as e:
            print("Rate limit error! Retrying...")
            sleep(60)
            continue
    target = response["choices"][0]["text"]
    full_program = prompt + target
    return full_program


openai.api_key = os.getenv("OPENAI_API_KEY")
parser = argparse.ArgumentParser()
parser.add_argument("input_file", type=str)
parser.add_argument("output_file", type=str)
parser.add_argument("--delay-sec", type=int, default=20)
parser.add_argument("--prompt-file", type=str, required=True)
args = parser.parse_args()

input_file = args.input_file
output_file = args.output_file
prompt_file = args.prompt_file
delay_sec = args.delay_sec

with open(prompt_file) as r:
    original_prompt = r.read().strip()

raw_records = read_jsonl(input_file)
correct_cnt, all_cnt = 0.0, 0.0
output_records = []
for example in raw_records:
    procedure = example["drawing_procedure"]
    prev_state = None
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
            continue
        all_cnt += 1.0
        true_actions = calc_actions(prev_state, state, True, True)
        output_record["true_actions"] = true_actions
        prompt += " " + instruction + "\n"
        program = codex_predict(prompt)
        try:
            output_record["program"] = program
            exec(program)
            pred_actions = grid.get_actions()
            output_record["pred_actions"] = pred_actions
        except Exception as e:
            traceback.print_exc()
            print(correct_cnt, all_cnt, correct_cnt/all_cnt)
            sleep(delay_sec)
            output_records.append(output_record)
            continue
        correct_cnt += float(true_actions == pred_actions)
        print(correct_cnt, all_cnt, correct_cnt/all_cnt)
        program += "\n# Cleaning\ngrid.clean_actions()\n"
        prompt = program + "\n#"
        sleep(delay_sec)
        prev_state = state
        output_records.append(output_record)
with open(output_file, "w") as w:
    for r in output_records:
        w.write(json.dumps(r, ensure_ascii=False).strip() + "\n")
print(correct_cnt, all_cnt, correct_cnt/all_cnt)

