class DataFormatter:
    def __init__(self):
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
        self.conversation_template = lambda text, query: [
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

    def conversations_to_chat(self, tokenizer, sample):
        # Add system message to conversation
        conv = sample["conversations"]
        conv.insert(
            0,
            {"from": "system", "value": self.system_prompt},
        )

        # Apply chat template by tokenizer
        chat = tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=False
        )
        sample["text"] = chat.strip()

        return sample
