from tqdm import tqdm

import argparse


class NEREvaluator:
    def __init__(self, ner_pipe, eval_dataset):
        self.ner_pipe = ner_pipe
        self.eval_dataset = eval_dataset

    def run(self):
        # Run predictions
        preds = []
        for sample in tqdm(eval_ds.dataset):
            out = predict(
                sample["input"],
                self.model,
                self.tokenizer,
                self.eval_ds.parse_output,
                "parsed",
            )
            preds.append(out)

        # Run evaluation
        labels = [
            self.eval_ds.parse_output(example["label"]) for example in eval_ds.dataset
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


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id")
    parser.add_argument("--max_length", default="512")

    args = parser.parse_args()
    model_id = args.model_id
    max_length = int(args.max_length)

    # Load model
    NER_model = NERModel("evaluation")
    NER_model.from_pretrained(model_id)

    NER_tokenizer = NERTokenizer()
    NER_tokenizer.from_pretrained(model_id)

    # Prepare data
    eval_ds = NERDataset()
    eval_ds.load_dataset(
        path="json", data_files="eval/test_data/CrossNER_AI.json", split="train"
    )
    eval_ds.convert_dataset("conversations", "instruction")

    # Create evaluator
    evaluator = NEREvaluator(NER_model.model, NER_tokenizer.tokenizer, eval_ds)

    # Run evaluation
    eval_result = evaluator.run()
    print(
        f'Precision: {eval_result["precision"]}, Recall: {eval_result["recall"]}, F1: {eval_result["f1"]}'
    )
