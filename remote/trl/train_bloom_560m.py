from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM
from transformers import BloomTokenizerFast
from peft import LoraConfig
from transformers import TrainingArguments

def convert_conversations(example):
    example["conversations"] = [{"role": conv["from"], "content": conv["value"]} for conv in example["conversations"]]
    return example

def main():
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom")
    raw_dataset = load_dataset('Universal-NER/Pile-NER-type', split='train')

    converted_data = raw_dataset.map(convert_conversations)

    print('Preparing dataset...')

    dataset = converted_data.map(
        lambda x: {
            'text': tokenizer.apply_chat_template(
                x['conversations'],
                tokenize=False,
                add_generation_prompt=False
            )
        }
    )

    print('Dataset is ready. Setup training configuration...')

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
    )

    training_args = TrainingArguments(
        output_dir='sft_bloom-560m_Pile-NER-type',
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
        model="bigscience/bloom-560m",
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
