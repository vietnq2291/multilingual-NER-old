from ner.pipeline import NERPipeline
import argparse


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipe_config_id", default=None)
    parser.add_argument("--max_length", default="512")

    args = parser.parse_args()
    pipe_config_id = args.pipe_config_id
    max_length = int(args.max_length)

    # deine pipeline
    ner_pipe = NERPipeline(pipe_config_id=pipe_config_id, usage="inference")

    # Inference loop
    while True:
        text = input("Text: ")
        if text == "":
            break
        entity_type = input("Entity type: ")
        if entity_type == "":
            break

        prompt = ner_pipe.data_formatter.gen_data_with_format(
            text=text, entity_type=entity_type
        )
        pred = ner_pipe.predict(prompt, max_length)
        print("Output:", pred)
        print("----------------------------------")


if __name__ == "__main__":
    main()
