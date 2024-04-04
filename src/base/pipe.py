from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
)

from src.utils import DataFormatter

import torch
import re


class NERPipeline:
    def __init__(self, usage="train"):
        if usage == "train":
            pass
        elif usage == "inference":
            pass
        elif usage == "evaluation":
            pass
        else:
            raise ValueError("Invalid pipeline usage")
        self.data_formatter = DataFormatter()

    def load_pretrained(self, pipeline_type, model_path, tokenizer_path=None):
        if not tokenizer_path:
            tokenizer_path = model_path
        self.pipeline_type = pipeline_type

        if pipeline_type == "mt5-instructions":
            self._prepare_mt5_instrucions(model_path, tokenizer_path)

    def forward(
        self,
        text=None,
        entity_type=None,
        format="raw",
        max_length=512,
        output_style="parsed",
    ):
        if format == "raw":
            inp = self.data_formatter.instruction_template["input"](
                text, self.data_formatter.query_template(entity_type)
            )
        else:
            inp = None

        input_tokenized = self.tokenizer(inp, add_special_tokens=True)
        input_ids = (
            torch.tensor(input_tokenized["input_ids"]).unsqueeze(0).to(self.device)
        )
        attention_mask = (
            torch.tensor(input_tokenized["attention_mask"]).unsqueeze(0).to(self.device)
        )

        output_ids = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_length=max_length
        )[0]
        output = self.tokenizer.decode(output_ids)

        if output_style == "parsed":
            output = self._parse_output(output)
        elif output_style == "raw":
            pass
        else:
            raise ValueError("Invalid style value!")

        return output

    def _parse_output(self, output):
        if "instructions" in self.pipeline_type:
            output = output.replace("[NEWLINE]", "\n")
            output = re.sub(r"<pad> ### Response:", "", output, 1)
            output = re.sub(r"</s>$", "", output)
            output = output.strip()

        try:
            output = eval(output)
        except:
            print("Warning: Output is not correctly formatted!")

        return output

    def _prepare_mt5_instrucions(self, model_path, tokenizer_path):
        model = MT5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = MT5Tokenizer.from_pretrained(tokenizer_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.tokenizer = tokenizer
