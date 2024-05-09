from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
import torch
from ner.pipeline import NERPipeline

output_dir = 'nqv2291/sft-mT5_base-InstructUIE_NER'
base_model_id = 'nqv2291/sft-mT5_base-NER'
dataset_path = 'nqv2291/en-InstructUIE_NER-train-instructions_format-mT5'
RANDOM_SEED = 42

pipe = NERPipeline('mt5-base-instructions', 'continue_train', base_model_id=base_model_id)

dataset = load_dataset(dataset_path)
dataset = dataset.shuffle(seed=RANDOM_SEED)

data_collator = DataCollatorForSeq2Seq(
    pipe.tokenizer, pad_to_multiple_of=8, return_tensors='pt', padding=True
)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    save_strategy='steps',
    evaluation_strategy="steps",
    report_to='wandb',
    save_total_limit=5,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=10,
    gradient_accumulation_steps=16,
    weight_decay=0.0,
    warmup_ratio=0.04,
    lr_scheduler_type='constant',
    learning_rate=2e-4,
    gradient_checkpointing=True,
    push_to_hub=True,
    bf16=True,
    tf32=True,
    optim='adafactor',
)

trainer = Trainer(
    model=pipe.model,
    tokenizer=pipe.tokenizer,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

trainer.train()
