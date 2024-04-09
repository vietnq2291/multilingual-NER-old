import sys
import json


def get_pipe_config(pipe_config_id):
    with open("ner/pipe_config.json", "r") as f:
        configs = json.load(f)
        pipe_config = configs[pipe_config_id]

    pipe_config["model_class"] = str_to_class(pipe_config["model_class"])
    pipe_config["tokenizer_class"] = str_to_class(pipe_config["tokenizer_class"])

    return configs[pipe_config]


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)
