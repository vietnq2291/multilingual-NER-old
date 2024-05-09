from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, Dataset
import torch
from ner.pipeline import NERPipeline

output_dir = 'nqv2291/sft-mT5_large-Pile-NER'
base_model_id = 'nqv2291/sft-mT5_large-alpaca'
dataset_path = 'nqv2291/en-Pile-NER-instructions_format-mT5'
RANDOM_SEED = 42

train_no = 4

pipe = NERPipeline('mt5-large-instructions', base_model_id=base_model_id, usage='continue_train')

# Get a range of dataset
dataset = load_dataset(dataset_path)
# dataset['train'] = dataset['train'].select(range(64000 * (train_no+1)))

##
print('Drop first 1k samples of segment (3->4)...')
no_skip_samples = 2000 
dataset['train'] = dataset['train'].select(
    list(range(64000*3)) + list(range(64000*3 + no_skip_samples, 64000 * (train_no+1) + no_skip_samples))
)
##

dataset = dataset.shuffle(seed=RANDOM_SEED)
print(f"Train the {train_no+1}th shard of dataset")
print("Number of training samples:", dataset['train'].shape[0])

data_collator = DataCollatorForSeq2Seq(
    pipe.tokenizer, pad_to_multiple_of=8, return_tensors='pt', padding=True
)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    save_strategy='steps',
    evaluation_strategy="steps",
    report_to='wandb',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=1000,
    eval_steps=500,
    logging_steps=10,
    gradient_accumulation_steps=16,
    warmup_ratio=0.04,
    lr_scheduler_type='constant',
    learning_rate=1e-4,
    gradient_checkpointing=True,
    push_to_hub=False,
    bf16=True,
    tf32=True,
    optim='adafactor',
    debug="underflow_overflow",
    seed=RANDOM_SEED,
)

trainer = Trainer(
    model=pipe.model,
    tokenizer=pipe.tokenizer,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

if train_no > 0:
    print('Continue training from latest checkpoint')
    trainer.train(resume_from_checkpoint=True)
else:
    print('Start training from base model')
    trainer.train(resume_from_checkpoint=False)
