class DataFormatter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        self.system_prompt = "A virtual assistant answers questions from a user based on the provided text."
        self.query_template = (
            lambda entity_type: f"What describes {entity_type} in the text?"
        )
        self.instruction_template = {
            "input": lambda text, query: (
                "[S2S] "
                + self.system_prompt
                + "\n\n### Instruction:\n"
                + query
                + "\n\n### Input:\n"
                + text
                + "\n\n<extra_id_0>"
            ).replace("\n", "[NEWLINE]"),
            "label": lambda target: ("### Response:\n" + target).replace(
                "\n", "[NEWLINE]"
            ),
        }
        self.conversation_template = {
            "input": lambda text, query: [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": f"Text: {text}",
                },
                {
                    "role": "assistant",
                    "content": "I've read this text.",
                },
                {
                    "role": "user",
                    "content": query,
                },
            ]
        }

    def format_output(self, output, data_style, model_config):
        if data_style == "instructions":
            output = output.replace("### Response:", "")
            output = output.replace("[NEWLINE]", "\n")
            output = output.strip()
        elif data_style in ["conversations", "sharegpt"]:
            if "LlamaForCausalLM" in model_config.architectures:
                output = output.split("[/INST]")[-1]
            pass
        else:
            raise ValueError("Invalid data style!")

        try:
            output = eval(output)
        except:
            output = []
        return output

    def gen_data_with_format(self, data_style, **kwargs):
        convert_fn = f"gen_{data_style}_data"
        return getattr(self, convert_fn)(**kwargs)

    def gen_instructions_data(self, **kwargs):
        if "conversations" in kwargs.keys():
            pass
        else:
            text = kwargs["text"]
            entity_type = kwargs["entity_type"]
            sample = self.conversation_template["input"](
                text, self.query_template(entity_type)
            )

        return sample

    def gen_conversations_data(self, **kwargs):
        if "conversations" in kwargs.keys():
            pass
        else:
            text = kwargs["text"]
            entity_type = kwargs["entity_type"]
            sample = self.instruction_template["input"](
                text, self.query_template(entity_type)
            )

        return sample

    def conversations_to_instructions(self, sample):
        text = sample["conversations"][0]["value"]
        query = sample["conversations"][2]["value"]
        target = sample["conversations"][-1]["value"]

        instruction_example = {
            "id": sample["id"],
            "input": self.instruction_template["input"](text, query),
            "label": self.instruction_template["label"](target),
        }

        return instruction_example

    def conversations_to_sharegpt(self, sample):
        sharegpt_conversations = []

        for turn in sample["conversations"]:
            if turn["from"] == "human":
                sharegpt_conversations.append(
                    {"role": "user", "content": turn["value"]}
                )
            elif turn["from"] == "gpt":
                sharegpt_conversations.append(
                    {"role": "assistant", "content": turn["value"]}
                )
            else:
                sharegpt_conversations.append(
                    {"role": turn["from"], "content": turn["value"]}
                )

        sharegpt_sample = {
            "id": sample["id"],
            "input": self.tokenizer.apply_chat_template(
                sharegpt_conversations[:-1], tokenize=False, add_generation_prompt=False
            ),
            "label": sharegpt_conversations[-1]["content"],
        }

        return sharegpt_sample
