import os
import torch
import csv
import json
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


def convert_csv_to_instruction_tuning(input_csv, output_json):
    result_json_list = []

    with open(input_csv, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            input_text = row['input_text']
            output_text = row['output_text']

            instruction_json = {
                "instruction": input_text,
                "output": output_text
            }

            result_json_list.append(instruction_json)

    with open(output_json, mode='w', encoding='utf-8') as json_file:
        json.dump(result_json_list, json_file, ensure_ascii=False, indent=4)


def peft_fine_tune(base_model_path, tokenizer_path, input_csv, output_json, num_epochs, batch_size):
    convert_csv_to_instruction_tuning(input_csv, output_json)

    dataset = load_dataset("json", data_files=output_json, split="train")

    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False
    )

    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    base_model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM"
    )

    training_params = TrainingArguments(
        output_dir="./Llama-3-8B-citeseer_nc",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_32bit",
        save_steps=2,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=100,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to=["tensorboard"]
    )

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=dataset,
        peft_config=peft_params,
        dataset_text_field="instruction",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )

    print("begin..............................................................!")
    trainer.train()
    trainer.save_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_path', type=str, default="./model/Llama-3-8B")
    parser.add_argument('--tokenizer_path', type=str, default="./model/Llama-3-8B")
    parser.add_argument('--input_csv', type=str, default="../dataset/citeseer_nc.csv")
    parser.add_argument('--output_json', type=str, default="../citeseer_nc.json")
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()

    peft_fine_tune(
        base_model_path=args.base_model_path,
        tokenizer_path=args.tokenizer_path,
        input_csv=args.input_csv,
        output_json=args.output_json,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size
    )
