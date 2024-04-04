from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    MT5ForConditionalGeneration,
    AutoTokenizer,
    MT5Tokenizer,
)

from src.utils import DataFormatter

import torch
import re


class NERPipeline:
    def __init__(self, usage="train"):
        self.usage = usage
        self.data_formatter = DataFormatter()

    def load_pretrained(self, pipeline_type, model_path, tokenizer_path=None):
        if not tokenizer_path:
            tokenizer_path = model_path
        self.pipeline_type = pipeline_type
        self.model_type, self.data_format = pipeline_type.split("-")

        if model_type == "mt5":
            self._prepare_mt5_instrucions(model_path, tokenizer_path)
        elif model_type == "tinyllama":
            self._prepare_tiny_llama_conversations(model_path, tokenizer_path)

        if data_format == "instructions":
            self.data_format_fn = lambda text, entity_type: (
                self.data_formatter.instruction_template["input"](
                    text, self.data_formatter.query_template(entity_type)
                )
            )
        elif data_format == "conversations":
            self.data_format_fn = lambda text, entity_type: (
                self.data_formatter.conversation_template(
                    text, self.data_formatter.query_template(entity_type)
                )
            )

        self._setup_usage()

    def forward(
        self,
        text=None,
        entity_type=None,
        format="raw",
        max_length=512,
        output_style="parsed",
    ):
        if format == "raw":
            inp = self.data_format_fn(text, entity_type)
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

    def _setup_usage(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        if self.usage == "inference" or self.usage == "evaluate":
            self.model.eval()

    def _parse_output(self, output):
        if self.data_format == "instructions":
            output = output.replace("[NEWLINE]", "\n")
            output = re.sub(r"<pad> ### Response:", "", output, 1)
            output = re.sub(r"</s>$", "", output)
            output = output.strip()
        elif self.data_format == "conversations":
            # Find the index of the last occurrence of the special token "</s>"
            end_idx = output.rfind("</s>")
            start_idx = output.rfind("[/INST]") + len("[/INST]")
            output = output[start_idx:end_idx]

        try:
            output = eval(output)
        except:
            print("Warning: Output is not correctly formatted!")

        return output

    def _prepare_mt5_instrucions(self, model_path, tokenizer_path):
        self.model = MT5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = MT5Tokenizer.from_pretrained(tokenizer_path)

    def _prepare_tiny_llama_conversations(self, model_path, tokenizer_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
