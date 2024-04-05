from peft import PeftModelForCausalLM
from transformers import AutoModelForCausalLM, BloomTokenizerFast, AutoTokenizer
from transformers import pipeline

import warnings
warnings.filterwarnings("ignore")

base_model_id = "bigscience/bloom-1b7"
adapter_id = "nqv2291/sft_bloom-1b7_Pile-NER-type"
#tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom")
tokenizer = AutoTokenizer.from_pretrained('bigscience/tokenizer')

#model_base = AutoModelForCausalLM.from_pretrained(base_model_id)
#model = PeftModelForCausalLM.from_pretrained(model_base, adapter_id)

model = AutoModelForCausalLM.from_pretrained('./sft_bloom-1b7_Pile-NER-type')
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def pred(text, entity_type):
    inp = [
        {'role': 'user', 'content': text},
        {'role': 'assistant', 'content': "I've read this text."},
        {'role': 'user', 'content': f"What describes {entity_type} in the text?"},
        {'role': "assistant", "content": ""}
    ]

    inp_formatted = tokenizer.apply_chat_template(
        inp,
        tokenize=False,
        add_generation_prompt=False,
        return_tensors='pt'
    )
    outputs = pipe(inp_formatted, max_new_tokens=256, do_sample=False, temperature=0.1, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)

    return inp_formatted, outputs[0]['generated_text'].split('</s>')[-1]

while True:
    text = input('text: ')
    if text == "":
        break
    entity_type = input('entity type: ')

    prompt, pred_entities = pred(text, entity_type)
    print('> prompt: ', prompt)
    print('> predict entities:', pred_entities, '\n')
