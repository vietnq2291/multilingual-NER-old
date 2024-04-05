from peft import PeftModelForCausalLM
from transformers import AutoModelForCausalLM, BloomTokenizerFast, AutoTokenizer
from transformers import pipeline

import warnings
warnings.filterwarnings("ignore")

base_model_id = "bigscience/bloom-1b7"
adapter_id = "nqv2291/sft_bloom-1b7_Pile-NER-type"
#tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom")
#tokenizer = AutoTokenizer.from_pretrained('bigscience/tokenizer')
tokenizer = AutoTokenizer.from_pretrained('PY007/TinyLlama-1.1B-step-50K-105b')

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

    return inp_formatted

while True:
    text = input('text: ')
    if text == "":
        break
    entity_type = input('entity type: ')

    prompt = pred(text, entity_type)
    print('> prompt: ', prompt)
