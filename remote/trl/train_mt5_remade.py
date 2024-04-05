from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer
from datasets import load_dataset
import argparse
from utils import (
    get_pretrained_tokenizer,
    get_pretrained_model,
)


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="google/mt5-base")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--model_output_dir", required=True)
    parser.add_argument("--push_to_hub", action="store_true", default=False)
    parser.add_argument("--random_seed", default=42)

    args = parser.parse_args()
    model_id = args.model_id
    dataset_path = args.dataset_path
    output_dir = args.dataset_output_dir
    is_push_to_hub = args.push_to_hub
    random_seed = args.random_seed

    # init model, tokenizer and dataset
    tokenizer = get_pretrained_tokenizer(model_id)
    model = get_pretrained_model(model_id)
    dataset = load_dataset(dataset_path)

    # Setup dataset
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    dataset = dataset.shuffle(seed=random_seed)

    # Setup training
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    dataset = dataset.shuffle(seed=42)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        gradient_accumulation_steps=16,
        weight_decay=0.0,
        warmup_ratio=0.04,
        lr_scheduler_type="cosine",
        learning_rate=2e-5,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        push_to_hub=is_push_to_hub,
        report_to="wandb",
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    # Start training
    model.train()
    trainer.train()

    print("Finished!")


if __name__ == "__main__":
    main()
