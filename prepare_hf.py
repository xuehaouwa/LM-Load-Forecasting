import pandas as pd
import os
import csv
import jsonlines


def clean(x):
    x = x.replace("\n", "")
    return x


def process(textified_file, save_folder, nlags=24):
    text_data = open(textified_file).readlines()
    num_train = int(len(text_data) * 0.7)
    num_test = int(len(text_data) * 0.2)
    num_vali = len(text_data) - num_train - num_test
    train_lines = text_data[:num_train]
    val_lines = text_data[num_train: num_train + num_vali]
    file_key = textified_file.split("/")[-1].split(".")[0]
    to_jsonl(train_lines, os.path.join(save_folder, file_key + "_train.json"), nlags)
    to_jsonl(val_lines, os.path.join(save_folder, file_key + "_val.json"), nlags)


def to_jsonl(text_lines, save_path, nlags):
    input_line = []
    outputs = []
    num_instance = len(text_lines) - nlags
    for i in range(num_instance):
        inputs = text_lines[i: i + nlags]
        input_line.append(" ".join(clean(t) for t in inputs))
        outputs.append(clean(text_lines[i + nlags]))

    json_items = []
    for j in range(num_instance):
        json_items.append({"text": input_line[j], "summary": outputs[j]})
    with jsonlines.open(save_path, 'w') as writer:
        writer.write_all(json_items)


def test_to_jsonl(text_lines, save_path, nlags, horizon):
    inputs = []
    outputs = []
    num_instance = len(text_lines) - nlags - horizon + 1
    for i in range(num_instance):
        in_lines = text_lines[i: i + nlags]
        targets = text_lines[i + nlags: i + nlags + horizon]
        inputs.append([clean(t) for t in in_lines])
        outputs.append([clean(t) for t in targets])

    json_items = []
    for j in range(num_instance):
        json_items.append({"input": inputs[j], "target": outputs[j]})
    with jsonlines.open(save_path, 'w') as writer:
        writer.write_all(json_items)


def process_testing(textified_file, save_folder, nlags=24, horizon=4):
    text_data = open(textified_file).readlines()
    num_train = int(len(text_data) * 0.7)
    num_test = int(len(text_data) * 0.2)
    num_vali = len(text_data) - num_train - num_test
    test_lines = text_data[num_train + num_vali:]
    file_key = textified_file.split("/")[-1].split(".")[0]
    test_to_jsonl(test_lines, os.path.join(save_folder, file_key + "_test.json"), nlags, horizon)


if __name__ == "__main__":
    process(textified_file="building_A.txt",
            save_folder="",
            nlags=24)

    process_testing(textified_file="building_A.txt",
                    save_folder="",
                    nlags=24, horizon=24)
