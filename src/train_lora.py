import os
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
DATASET_PATH = os.getenv("DATASET_PATH", "data/support_data.jsonl")
ADAPTER_DIR = os.getenv("ADAPTER_DIR", "resolve-lora-adapter")
FINAL_MODEL_DIR = os.getenv("FINAL_MODEL_DIR", "resolve-final-model")


def train_and_merge() -> None:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    def format_chat(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    formatted_dataset = dataset.map(format_chat)

    training_args = SFTConfig(
        output_dir="./lora-checkpoints",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=5,
        save_strategy="no",
        max_length=512,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        args=training_args,
        processing_class=tokenizer,
    )
    trainer.train()

    trainer.model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    del model, trainer
    torch.cuda.empty_cache()

    base_fp16 = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    merged_model = PeftModel.from_pretrained(base_fp16, ADAPTER_DIR).merge_and_unload()
    merged_model.save_pretrained(FINAL_MODEL_DIR, safe_serialization=True)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)


if __name__ == "__main__":
    train_and_merge()