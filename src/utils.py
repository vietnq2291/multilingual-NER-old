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
