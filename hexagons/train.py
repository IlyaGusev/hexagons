import argparse
import random
import json

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, logging

from hexagons.dataset import DATASET_CLASSES, read_records
from hexagons.util import set_random_seed, fix_tokenizer, fix_model
from hexagons.common import MODEL_CLASSES


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
    model_type = config.get("model_type", "seq2seq")
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, truncation_side="left")
    if model_type == "causal":
        tokenizer = fix_tokenizer(tokenizer)

    is_one_based = config.get("is_one_based", True)
    use_color_strings = config.get("use_color_strings", False)
    input_template = config.get("input_template", None)
    train_records = read_records(
        train_file,
        is_one_based=is_one_based,
        use_color_strings=use_color_strings,
        input_template=input_template
    )
    val_records = read_records(
        val_file,
        is_one_based=is_one_based,
        use_color_strings=use_color_strings,
        input_template=input_template
    )
    random.shuffle(train_records)
    print(train_records[0])

    max_source_tokens_count = config["max_source_tokens_count"]
    max_target_tokens_count = config["max_target_tokens_count"]
    dataset_class = DATASET_CLASSES[model_type]
    train_dataset = dataset_class(
        train_records,
        tokenizer,
        max_source_tokens_count=max_source_tokens_count,
        max_target_tokens_count=max_target_tokens_count
    )
    print(train_dataset[0])

    val_dataset = dataset_class(
        val_records,
        tokenizer,
        max_source_tokens_count=max_source_tokens_count,
        max_target_tokens_count=max_target_tokens_count
    )

    model_class = MODEL_CLASSES[model_type]
    model = model_class.from_pretrained(model_name)
    model = fix_model(model, tokenizer, model_type, max_target_tokens_count)

    # Default model generation params
    model.config.num_beams = 5
    max_tokens_count = max_target_tokens_count + max_source_tokens_count
    model.config.max_length = max_target_tokens_count if model_type == "seq2seq" else max_tokens_count

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
