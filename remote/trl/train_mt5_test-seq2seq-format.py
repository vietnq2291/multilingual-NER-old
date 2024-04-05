from transformers import AutoTokenizer, T5Tokenizer,T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
import torch

model_name = 'google/mt5-base'
output_dir = 'sft_mt5-Pile-NER-test-seq2seq-format'
tokenizer_chat_template = """
{% for message in messages %}
{% if message['from'] == 'human' %}
{{'USER: ' + message['value']}}
{% elif message['from'] == 'gpt' %}
{{'ASSISTANT: ' + message['value'] + '</s>'}}
{% else %}{{message['value']}}
{% endif %}{% endfor %}"""
max_length = 512

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

def make_labelled_data(examples):
    ids, texts, labels = [], [], []

    for example_id, example_text in zip(examples['id'], examples['text']):
        paragraphs = example_text.split('</s>')
        system_prompt, text = paragraphs[0].split('\nUSER: ')
        text = text[:text.find('\nASSISTANT:')]
        dialogs = paragraphs[1:]

        cnt = 0
        for dialog in dialogs:
            if len(dialog) > 0:
                sep_idx_user = dialog.find('USER: ') + len('USER: ')-1
                sep_idx_assistant = dialog.find('\nASSISTANT: ') + len('\nASSISTANT: ')-1
                user_question = dialog[sep_idx_user:dialog.find('\nASSISTANT: ')]
                assistant_answer = dialog[sep_idx_assistant:].strip()

                new_id = str(example_id) + '_' + str(cnt+1)
                new_inp = ('[S2S] input: ' + system_prompt + ' \n ' + user_question + ' \n '  + text + ' <extra_id_0>').replace('\n', '[NEWLINE]')
                new_label = 'output: ' + assistant_answer

                ids.append(new_id)

                texts.append(new_inp)
                labels.append(new_label)
                cnt += 1
    return {'id': ids, 'text': texts, 'label': labels}

def tokenize(text, tokenizer, input_field, label_field):
    tmp = tokenizer(text[input_field], truncation=True,max_length =max_length,padding=False,add_special_tokens=False)
    text['input_ids'] = tmp['input_ids']
    text['attention_mask'] = tmp['attention_mask']
    for i in range(len(text['input_ids'])):
        if text['input_ids'][i][-1] != tokenizer.eos_token_id and \
            len(text['input_ids'][i]) < max_length:
                text['input_ids'][i].append(tokenizer.eos_token_id)
                text['attention_mask'][i].append(1)


    text['labels'] = tokenizer(text[label_field], truncation=True, max_length=max_length,padding=False)['input_ids']
    for i in range(len(text['labels'])):
        if text['labels'][i][-1] != tokenizer.eos_token_id and \
            len(text['labels'][i]) < max_length:
                text['labels'][i].append(tokenizer.eos_token_id)

    assert all([text['input_ids'][i][-1] == tokenizer.eos_token_id or len(text['input_ids'][i]) == max_length for i in range(len(text))]), str([text['input_ids'][i][-1] for i in range(len(text))])+str([len(text['input_ids'][i]) for i in range(len(text))])
    return text

def main():
    # Define model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    tokenizer.eos_token = '<\s>'
    tokenizer.bos_token = tokenizer.pad_token
    tokenizer.sep_token = '<s>'
    tokenizer.chat_template = tokenizer_chat_template

    # Prepare data
    print('Preparing dataset...')
    ## Load dataset and re-format chat style
    raw_dataset = load_dataset('Universal-NER/Pile-NER-type', split='train')
    raw_dataset = raw_dataset.select(range(2000))
    dataset = raw_dataset.map(lambda x: format_chat_template(tokenizer, x), remove_columns='conversations')
    dataset = dataset.train_test_split(test_size=0.02, seed=42)


    ## Making labelled dataset
    dataset['train'] = dataset['train'].map(make_labelled_data, batched=True, remove_columns=dataset['train'].column_names)
    dataset['test'] = dataset['test'].map(make_labelled_data, batched=True, remove_columns=dataset['test'].column_names)
    ## Tokenize dataset
    data = dataset.map(lambda x: tokenize(x, tokenizer, 'text', 'label'), batched=True,remove_columns=['text','label'])
    print('Dataset is ready. Setup training configuration...')

    # Setting training configurations
    data_collator = DataCollatorForSeq2Seq(tokenizer,pad_to_multiple_of=8, return_tensors='pt',padding=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        save_strategy='steps',
        report_to='wandb',
        save_total_limit=4,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        # save_steps=1000,
        # eval_steps=1000,
        save_steps=400,
        eval_steps=400,
        logging_steps=100,
        gradient_accumulation_steps=16,
        weight_decay=0.0,
        warmup_ratio=0.04,
        lr_scheduler_type='cosine',
        learning_rate=2e-5,
        gradient_checkpointing=True,
        push_to_hub=True,
        load_best_model_at_end=True,
        bf16=True,
        tf32=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=data['train'],
        eval_dataset=data['test'],

    )
    # model.config.use_cache = False
    model.train()
    trainer.train()
    model.save_pretrained(output_dir)


if __name__ == '__main__':
    main()
