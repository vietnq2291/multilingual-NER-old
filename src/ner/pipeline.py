import torch


class NERPipeline:
    def __init__(self, data_formatter, tokenizer, model, usage="inference"):
        self.data_formatter = data_formatter
        self.tokenizer = tokenizer
        self.model = model
        self.usage = usage
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
            output = self.data_formatter.format_output(output, data_style)
        return output

    def _setup_usage(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        if self.usage == "inference" or self.usage == "evaluate":
            self.model.eval()
        elif self.usage == "train":
            self.model.train()
