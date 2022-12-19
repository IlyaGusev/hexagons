import argparse
import traceback
import json
from time import sleep

from tqdm import tqdm

from hexagons.codex import codex_predict


parser = argparse.ArgumentParser()
parser.add_argument("input_file", type=str)
parser.add_argument("--delay-sec", type=int, default=15)
parser.add_argument("--prompt-file", type=str, required=True)
args = parser.parse_args()

with open(args.prompt_file) as r:
    original_prompt = r.read().strip()

with open(args.input_file) as r:
    records = json.load(r)["examples"]

correct_cnt = 0.0
all_cnt = 0.0
for r in tqdm(records):
    all_cnt += 1.0
    instruction = r["input"]
    target = r["target_scores"]["True"]
    prompt = original_prompt + " " + instruction + "\n"
    program = codex_predict(prompt)
    print(program)
    prediction = int(program.strip().split()[-1] == "True")
    correct_cnt += float(prediction == target)
    print("Pred:", prediction)
    print("Target:", target)
    #try:
    #    exec(program)
    #    print("X:", coords.x)
    #    print("Y:", coords.y)
    #    prediction = int(coords.x == 0 and coords.y == 0)
    #    print("Pred:", prediction)
    #    print("Target:", target)
    #    correct_cnt += float(prediction == target)
    #except Exception as e:
    #    traceback.print_exc()
    print(correct_cnt, all_cnt, correct_cnt/all_cnt)
    sleep(args.delay_sec)
