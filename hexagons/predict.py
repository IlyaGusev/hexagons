import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from hexagons.dataset import convert_to_t2t
from hexagons.util import read_jsonl, gen_batch, set_random_seed


def predict(
    nrows,
    model_name,
    input_file,
    output_file,
    batch_size,
    max_source_tokens_count,
    seed,
    num_beams
):
    set_random_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        do_lower_case=False,
        truncation_side="left"
    )
    raw_records = read_jsonl(input_file)
    records = []
    for example in raw_records:
        records.extend(convert_to_t2t(example["drawing_procedure"]))
    if nrows:
        records = records[:nrows]

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)


    predictions = []
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
            num_beams=num_beams
        )

        for ids in output_ids:
            prediction = tokenizer.decode(ids, skip_special_tokens=True)
            predictions.append(prediction)

    with open(output_file, "w") as w:
        for p in predictions:
            w.write(p.strip() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-tokens-count", type=int, default=512)
    parser.add_argument("--num-beams", type=int, default=5)
    args = parser.parse_args()
    predict(**vars(args))
