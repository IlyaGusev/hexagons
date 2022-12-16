import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from hexagons.board import COLORS_MAPPING
from hexagons.dataset import read_records
from hexagons.util import gen_batch, set_random_seed
from hexagons.common import MODEL_CLASSES


def predict(
    nrows,
    model_name,
    input_file,
    output_file,
    batch_size,
    max_source_tokens_count,
    max_target_tokens_count,
    model_type,
    seed,
    num_beams,
    is_zero_based,
    use_color_strings,
    input_template
):
    set_random_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        do_lower_case=False,
        truncation_side="left"
    )
    records = read_records(
        input_file,
        not is_zero_based,
        use_color_strings,
        input_template=input_template
    )
    if nrows:
        records = records[:nrows]

    model = MODEL_CLASSES[model_type].from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    predictions = []
    if model_type == "seq2seq":
        for batch in tqdm(gen_batch(records, batch_size)):
            texts = [r["source"] for r in batch]
            input_ids = tokenizer(
                texts,
                add_special_tokens=True,
                max_length=max_source_tokens_count,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )["input_ids"].to(device)
            output_ids = model.generate(
                input_ids=input_ids,
                num_beams=num_beams,
                max_length=max_target_tokens_count
            )

            for ids in output_ids:
                prediction = tokenizer.decode(ids, skip_special_tokens=True)
                predictions.append(prediction)
    else:
        for r in tqdm(records):
            text_tokens = tokenizer(
                r["source"],
                add_special_tokens=False,
                max_length=max_source_tokens_count,
                padding=False,
                truncation=True
            )["input_ids"]
            input_ids = text_tokens + [tokenizer.sep_token_id]
            input_ids = torch.LongTensor([input_ids]).to(device)
            output_ids = model.generate(
                input_ids=input_ids,
                num_beams=num_beams,
                max_new_tokens=max_target_tokens_count
            )
            prediction = tokenizer.decode(output_ids[0], skip_special_tokens=False)
            prediction = prediction.split(tokenizer.sep_token)[1].split(tokenizer.eos_token)[0]
            predictions.append(prediction)

    em_correct = 0.0
    em_all = 0.0
    for r, pred in zip(records, predictions):
        true_actions = r["actions"]
        pred_actions = pred.split(" end")[0].split(",")
        cleaned_pred_actions = []
        for action in pred_actions:
            try:
                parts = tuple(action.split())
                assert len(parts) == 3
                cleaned_action = (
                    int(parts[0]),
                    int(parts[1]),
                    int(parts[2]) if not use_color_strings else parts[2]
                )
                if use_color_strings:
                    assert cleaned_action[2] in list(COLORS_MAPPING.values())
                else:
                    assert cleaned_action[2] < 8
                assert cleaned_action[0] <= 10
                assert cleaned_action[1] <= 18
                cleaned_pred_actions.append(cleaned_action)
            except Exception as e:
                continue
        pred_actions = cleaned_pred_actions
        true_actions.sort()
        pred_actions.sort()
        em_correct += float(true_actions == pred_actions)
        em_all += 1.0
    print(em_correct/em_all)

    with open(output_file, "w") as w:
        for p in predictions:
            w.write(p.strip() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-type", type=str, default="seq2seq", choices=("seq2seq", "causal"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-source-tokens-count", type=int, default=512)
    parser.add_argument("--max-target-tokens-count", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--is-zero-based", action="store_true", default=False)
    parser.add_argument("--use-color-strings", action="store_true", default=False)
    parser.add_argument("--input-template", type=str, default="instructions: {} commands: ")
    args = parser.parse_args()
    predict(**vars(args))
