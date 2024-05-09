from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, concatenate_datasets
import torch
from ner.pipeline import NERPipeline

output_dir = 'mt5_small_alpaca-0'
base_model_id = 'google/mt5-small'
dataset_path = 'nqv2291/en-alpaca-instructions_format-mT5'
RANDOM_SEED = 42

pipe = NERPipeline('mt5-small-instructions', base_model_id=base_model_id, usage='train')

dataset = load_dataset(dataset_path, split='train')
dataset = dataset.select([0,1,2,3])

for x in dataset:
    print(x['id'])
print('-----')
dataset = dataset.shuffle(seed=RANDOM_SEED)



data_collator = DataCollatorForSeq2Seq(
    pipe.tokenizer, pad_to_multiple_of=8, return_tensors='pt', padding=True
)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    save_strategy='steps',
    report_to=None,
    save_total_limit=3,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    logging_steps=1,
    save_steps=2,
    lr_scheduler_type='constant',
    learning_rate=1e-5,
    push_to_hub=True,
    bf16=True,
    tf32=True,
    optim='adafactor',
    seed=42,
)

trainer = Trainer(
    model=pipe.model,
    tokenizer=pipe.tokenizer,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

#trainer.train(resume_from_checkpoint=False)
trainer.train(resume_from_checkpoint=True)
