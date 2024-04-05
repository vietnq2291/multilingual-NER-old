from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import torch


MAX_LENGTH = 512
TOKENIZER_CHAT_TEMPLATE = """
{% for message in messages %}
{% if message['from'] == 'human' %}
{{'USER: ' + message['value']}}
{% elif message['from'] == 'gpt' %}
{{'ASSISTANT: ' + message['value'] + '</s>'}}
{% else %}{{message['value']}}
{% endif %}{% endfor %}"""


def get_pretrained_model(model_id):
    model = MT5ForConditionalGeneration.from_pretrained(model_id)
    return model


def get_pretrained_tokenizer(model_id):
    tokenizer = MT5Tokenizer.from_pretrained(model_id)

    tokenizer.bos_token = tokenizer.pad_token
    tokenizer.sep_token = tokenizer.eos_token
    tokenizer.chat_template = TOKENIZER_CHAT_TEMPLATE

    return tokenizer


def format_chat_template(tokenizer, example):
    # Add system message to conversation
    example_text = example["conversations"]
    example_text.insert(
        0,
        {
            "from": "system",
            "value": "A virtual assistant answers questions from a user based on the provided paragraph.",
        },
    )

    # Apply chat template by tokenizer
    example_text_chat_template = tokenizer.apply_chat_template(
        example_text, tokenize=False, add_generation_prompt=False
    )
    example["text"] = example_text_chat_template.strip()

    return example


def make_labelled_data(examples):
    ids, texts, labels = [], [], []

    for example_id, example_text in zip(examples["id"], examples["text"]):
        paragraphs = example_text.split("</s>")
        system_prompt, text = paragraphs[0].split("\nUSER: ")
        text = text[: text.find("\nASSISTANT:")]
        dialogs = paragraphs[1:]

        cnt = 0
        for dialog in dialogs:
            if len(dialog) > 0:
                sep_idx_user = dialog.find("USER: ") + len("USER: ") - 1
                sep_idx_assistant = (
                    dialog.find("\nASSISTANT: ") + len("\nASSISTANT: ") - 1
                )
                user_question = dialog[sep_idx_user : dialog.find("\nASSISTANT: ")]
                assistant_answer = dialog[sep_idx_assistant:].strip()

                new_id = str(example_id) + "_" + str(cnt + 1)
                new_inp = (
                    "[S2S] input: "
                    + system_prompt
                    + " \n "
                    + user_question
                    + " \n "
                    + text
                    + " <extra_id_0>"
                ).replace("\n", "[NEWLINE]")
                new_label = "output: " + assistant_answer

                ids.append(new_id)

                texts.append(new_inp)
                labels.append(new_label)
                cnt += 1
    return {"id": ids, "text": texts, "label": labels}


def tokenize(text, tokenizer, input_field, label_field, max_length=MAX_LENGTH):
    tmp = tokenizer(
        text[input_field],
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )
    text["input_ids"] = tmp["input_ids"]
    text["attention_mask"] = tmp["attention_mask"]
    for i in range(len(text["input_ids"])):
        if (
            text["input_ids"][i][-1] != tokenizer.eos_token_id
            and len(text["input_ids"][i]) < max_length
        ):
            text["input_ids"][i].append(tokenizer.eos_token_id)
            text["attention_mask"][i].append(1)

    text["labels"] = tokenizer(
        text[label_field],
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )["input_ids"]
    for i in range(len(text["labels"])):
        if (
            text["labels"][i][-1] != tokenizer.eos_token_id
            and len(text["labels"][i]) < max_length
        ):
            text["labels"][i].append(tokenizer.eos_token_id)

    assert all(
        [
            text["input_ids"][i][-1] == tokenizer.eos_token_id
            or len(text["input_ids"][i]) == max_length
            for i in range(len(text))
        ]
    ), str([text["input_ids"][i][-1] for i in range(len(text))]) + str(
        [len(text["input_ids"][i]) for i in range(len(text))]
    )
    return text
