from transformers import (
    AutoTokenizer,
    MT5Tokenizer,
    AutoModelForCausalLM,
    MT5ForConditionalGeneration,
)
import torch

from ner.data_formatter import DataFormatter
from ner.utils import get_pipe_config


class NERPipeline:
    def __init__(self, pipe_config_id, usage="inference"):
        self.pipe_config_id = pipe_config_id
        self.usage = usage
        self._load_pipe_from_config()
        self._setup_usage()

    def predict(self, inp, max_length, data_style=None):
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
        output = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        if data_style:
            output = self.data_formatter.format_output(output, data_style, self.model.config)
        return output

    def _load_pipe_from_config(self):
        pipe_config = get_pipe_config(self.pipe_config_id)
        self.model = pipe_config["model_class"].from_pretrained(pipe_config["model_id"])
        self.tokenizer = pipe_config["tokenizer_class"].from_pretrained(
            pipe_config["model_id"]
        )
        self.data_formatter = DataFormatter(self.tokenizer)

    def _setup_usage(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        if self.usage == "inference" or self.usage == "evaluate":
            self.model.eval()
        elif self.usage == "train":
            self.model.train()
