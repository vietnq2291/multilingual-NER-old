from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM
from transformers import BloomTokenizerFast
from peft import LoraConfig
from transformers import TrainingArguments


def format_chat_template(tokenizer, example):
    # Format roles
    example_text = example['conversations']
    for i in range(len(example_text)):
        if example_text[i]['from'] == 'human':
            role = 'user'
        elif example_text[i]['from'] == 'gpt':
            role = 'assistant'
        else:
            raise ValueError(f'''Unexpected role: "{example_text[i]['from']}"''')
        example_text[i] = {
            'role': role,
            'content': example_text[i]['value']
        }

    # Apply chat template by tokenizer
    example_text_chat_template = tokenizer.apply_chat_template(
        example_text,
        tokenize=False,
        add_generation_prompt=False
    )
    example['text'] = example_text_chat_template

    return example

def main():
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom")

    print('Preparing dataset...')
    raw_dataset = load_dataset('Universal-NER/Pile-NER-type', split='train')
    dataset = raw_dataset.map(lambda x: format_chat_template(tokenizer, x), remove_columns='conversations')
    print('Dataset is ready. Setup training configuration...')

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
    )

    training_args = TrainingArguments(
        output_dir='sft_bloom-1b7_Pile-NER-type',
        report_to='wandb',
        learning_rate=2e-5,
        weight_decay=0,
        warmup_ratio=0.04,
        lr_scheduler_type='cosine',
        bf16=True,
        tf32=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        logging_steps=100,
        save_steps=500,
        save_total_limit=3,
        num_train_epochs=3,
        max_steps=-1,
        push_to_hub=True,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model="bigscience/bloom-1b7",
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        packing=True,
        peft_config=peft_config
    )

    print('Start training...')
    trainer.train()
    print('Finished training.')

if __name__ == "__main__":
    main()
