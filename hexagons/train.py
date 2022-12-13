import argparse
import random
import json

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Trainer, TrainingArguments, logging

from hexagons.dataset import T2TDataset, convert_to_t2t
from hexagons.util import read_jsonl, set_random_seed


def train(
    config_file,
    checkpoint,
    train_file,
    val_file,
    train_sample_rate,
    val_sample_rate,
    output_dir,
    report_to,
    seed
):
    set_random_seed(seed)
    logging.set_verbosity_info()
    with open(config_file, "r") as r:
        config = json.load(r)

    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, truncation_side="left")

    train_raw = read_jsonl(train_file)
    train_records = []
    for example in train_raw:
        train_records.extend(convert_to_t2t(example["drawing_procedure"]))
    random.shuffle(train_records)

    val_raw = read_jsonl(val_file)
    val_records = []
    for example in val_raw:
        val_records.extend(convert_to_t2t(example["drawing_procedure"]))

    max_source_tokens_count = config["max_source_tokens_count"]
    max_target_tokens_count = config["max_target_tokens_count"]
    train_dataset = T2TDataset(
        train_records,
        tokenizer,
        max_source_tokens_count=max_source_tokens_count,
        max_target_tokens_count=max_target_tokens_count
    )

    val_dataset = T2TDataset(
        val_records,
        tokenizer,
        max_source_tokens_count=max_source_tokens_count,
        max_target_tokens_count=max_target_tokens_count
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.config.num_beams = 5
    model.config.max_length = max_target_tokens_count

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        logging_steps=config["logging_steps"],
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        num_train_epochs=config["num_train_epochs"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        evaluation_strategy="steps",
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to=report_to
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--val-file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-sample-rate", type=float, default=1.0)
    parser.add_argument("--val-sample-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", type=str, default="none")
    args = parser.parse_args()
    train(**vars(args))
