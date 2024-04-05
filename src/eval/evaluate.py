from datasets import load_dataset
from src.base.pipe import NERPipeline
from tqdm import tqdm

import argparse


class NEREvaluator:
    def __init__(self, ner_pipe, eval_data_path):
        self.ner_pipe = ner_pipe
        self._load_dataset(eval_data_path)

    def run(self, max_length):
        # Run predictions
        preds = []
        for sample in tqdm(self.dataset):
            out = ner_pipe.forward(
                sample=sample["input"],
                max_length=max_length,
                output_style="parsed",
            )
            preds.append(out)

        # Run evaluation
        labels = [
            self.dataset.parse_output(example["label"]) for example in self.dataset
        ]
        return self.evaluate(preds, labels)

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

    def _load_dataset(self, eval_data_path):
        # Prepare data
        dataset = load_dataset(path="json", data_files=eval_data_path, split="train")
        if self.ner_pipe.data_format == "instructions":
            dataset = dataset.map(
                self.ner_pipe.data_formatter.conversations_to_instructions
            )
        elif self.ner_pipe.data_format == "conversations":
            pass
        self.dataset = dataset


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--tokenizer_id", default=None)
    parser.add_argument("--pipeline_type", default=None)
    parser.add_argument("--max_length", default="512")
    parser.add_argument("--eval_data_path")

    args = parser.parse_args()
    model_id = args.model_id
    tokenizer_id = args.tokenizer_id
    pipeline_type = args.pipeline_type
    max_length = int(args.max_length)
    eval_data_path = args.eval_data_path

    ner_pipe = NERPipeline(usage="evaluate")
    ner_pipe.load_pretrained(pipeline_type, model_id, tokenizer_id)

    # Create evaluator
    evaluator = NEREvaluator(ner_pipe, eval_data_path)

    # Run evaluation
    eval_result = evaluator.run(max_length)
    print(
        f'Precision: {eval_result["precision"]}, Recall: {eval_result["recall"]}, F1: {eval_result["f1"]}'
    )
