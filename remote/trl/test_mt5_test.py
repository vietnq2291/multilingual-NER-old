from transformers import AutoTokenizer, T5Tokenizer,T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
import torch


model_name = 'google/mt5-base'
output_dir = 'sft_mt5-Pile-NER-test'
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
        text = ''.join(paragraphs[:1])
        dialogs = paragraphs[1:]

        cnt = 0
        for dialog in dialogs:
            if len(dialog) > 0:
                sep_str = '\nASSISTANT: '
                sep_idx = dialog.find(sep_str) # + len(sep_str)-1

                ids.append(str(example_id) + '_' + str(cnt+1))
                texts.append(('[S2S] ' + text + dialog[:sep_idx] + '<extra_id_0>').replace('\n', '[NEWLINE]'))
                labels.append(dialog[sep_idx:].strip())
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

def predict(sample, model, tokenizer, print_output=True):
    sample = '[S2S] ' + sample
    sample+= '<extra_id_0>'
    test = tokenizer(sample, add_special_tokens=False)
    input_ids = torch.tensor(test['input_ids']).unsqueeze(0).to('cuda')
    attention_mask  =torch.tensor(test['attention_mask']).unsqueeze(0).to('cuda')
    out = tokenizer.decode(model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=256, )[0])
    if print_output:
        print(sample.replace('[NEWLINE]','\n'))
        print(out.replace('[NEWLINE]','\n'))
    return out

def main():

    # Define model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained('/home/nlp/vietnq29/multilingual-ner/trl/sft_mt5-Pile-NER-test')
    model.eval()
    model.cuda()
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    tokenizer.eos_token = '<\s>'
    tokenizer.bos_token = tokenizer.pad_token
    tokenizer.sep_token = '<s>'
    tokenizer.chat_template = tokenizer_chat_template

    while True:
        inp = input('text: ')
        if inp == '': break
        entity = input('entity type: ')
        custom_sample = f"""A virtual assistant answers questions from a user based on the provided paragraph.
USER: Text: {inp}
ASSISTANT: I've read this text.
USER: What describes {entity} in the text?
"""
        res = predict(custom_sample, model, tokenizer, print_output=True)

if __name__=='__main__':
    main()
