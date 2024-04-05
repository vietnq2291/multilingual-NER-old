from datasets import load_dataset
from trl import SFTTrainer
from transformers import MT5ForConditionalGeneration, AutoTokenizer
from peft import LoraConfig
from transformers import TrainingArguments


model_id = 'google/mt5-base'
output_dir = 'sft_mt5-base_Pile-NER-type'
tokenizer_chat_template = """
{% for message in messages %}
{% if message['from'] == 'human' %}
{{'USER: ' + message['value']}}
{% elif message['from'] == 'gpt' %}
{{'ASSISTANT: ' + message['value'] + '</s>'}}
{% else %}{{message['value']}}
{% endif %}{% endfor %}"""



def format_chat_template(tokenizer, example):
    # Add system message to conversation
    example_text = example['conversations']
    example_text.insert(0, {'from': 'system', 'value': 'A virtual assistant answers questions from a user based on the provided paragraph.'})

    # Apply chat template by tokenizer
    example_text_chat_template = tokenizer.apply_chat_template(
        example_text,
        tokenize=False,
        add_generation_prompt=False
    )
    example['text'] = example_text_chat_template.strip()

    return example


def main():
    # define Model and Tokenizer
    model = MT5ForConditionalGeneration.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained('google/mt5-base')
    tokenizer.chat_template = tokenizer_chat_template

    # Prepare data
    print('Preparing dataset...')
    raw_dataset = load_dataset('Universal-NER/Pile-NER-type', split='train')
    dataset = raw_dataset.map(lambda x: format_chat_template(tokenizer, x), remove_columns='conversations')
    dataset = dataset.train_test_split(test_size=0.02, seed=42)
    print('Dataset is ready. Setup training configuration...')

    # peft_config = LoraConfig(
    #     r=64,
    #     lora_alpha=16,
    #     lora_dropout=0.05,
    #     bias="none",
    # )

    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to='wandb',
        learning_rate=2e-5,
        weight_decay=0,
        warmup_ratio=0.04,
        lr_scheduler_type='cosine',
        bf16=True,
        tf32=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=16,
        logging_steps=100,
        evaluation_strategy="steps",
        save_steps=1000,
        eval_steps=1000,
        save_total_limit=4,
        num_train_epochs=3,
        max_steps=-1,
        push_to_hub=True,
        gradient_checkpointing=True,
        load_best_model_at_end=True
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        dataset_text_field='text',
        max_seq_length=1024,
        packing=True,
        # peft_config=peft_config
    )

    print('Start training...')
    trainer.train()
    print('Finished training.')

if __name__ == "__main__":
    main()
