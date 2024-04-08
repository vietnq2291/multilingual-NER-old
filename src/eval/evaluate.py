from datasets import load_dataset
from tqdm import tqdm

import argparse


from tqdm import tqdm


class NEREvaluator:
    def __init__(self, data_formatter, pipeline, evaluate_data_path, data_style):
        self.data_formatter = data_formatter
        self.pipeline = pipeline
        self.data_style = data_style
        self._load_dataset(evaluate_data_path)

    def run(self, max_length):
        preds = []
        for sample in tqdm(self.dataset):
            pred = self.pipeline.predict(sample["input"], max_length, self.data_style)
            preds.append(pred)
        self.evaluate_results = self.evaluate(preds, self.dataset["label"])
        return self.evaluate_results

    def evaluate(self, preds, labels):
        n_correct, n_pos_label, n_pos_pred = 0, 0, 0
        for pred, label in zip(preds, labels):
            n_correct += sum([1 if entity in label else 0 for entity in pred])
            n_pos_pred += len(pred)
            n_pos_label += len(label)
        prec = n_correct / (n_pos_pred + 1e-10)
        recall = n_correct / (n_pos_label + 1e-10)
        f1 = 2 * prec * recall / (prec + recall + 1e-10)
        return {
            "precision": prec,
            "recall": recall,
            "f1": f1,
        }

    def _load_dataset(self, data_path):
        dataset = load_dataset(path="json", data_files=data_path, split="train")
        if self.data_style == "instructions":
            dataset = dataset.map(
                self.data_formatter.conversations_to_instructions,
                remove_columns=dataset.column_names,
            )

        dataset = dataset.map(
            lambda sample: {
                "label": self.data_formatter.format_output(
                    sample["label"], self.data_style
                )
            }
        )
        self.dataset = dataset
