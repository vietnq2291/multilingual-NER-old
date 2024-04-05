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
