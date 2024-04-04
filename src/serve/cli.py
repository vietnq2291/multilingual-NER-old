from src.base.pipe import NERPipeline
import argparse


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--tokenizer_id", default=None)
    parser.add_argument("--pipeline_type", default=None)
    parser.add_argument("--max_length", default="512")
    parser.add_argument("--output_style", default="parsed")

    args = parser.parse_args()
    model_id = args.model_id
    tokenizer_id = args.tokenizer_id
    pipeline_type = args.pipeline_type
    max_length = int(args.max_length)
    output_style = args.output_style

    ner_pipe = NERPipeline(usage="inference")
    ner_pipe.load_pretrained(pipeline_type, model_id, tokenizer_id)

    # Inference loop
    while True:
        text = input("Text: ")
        if text == "":
            break
        entity_type = input("Entity type: ")
        if entity_type == "":
            break
        out = ner_pipe.forward(
            text=text,
            entity_type=entity_type,
            max_length=max_length,
            output_style=output_style,
        )
        print("Output:", out)
        print("----------------------------------")


if __name__ == "__main__":
    main()
