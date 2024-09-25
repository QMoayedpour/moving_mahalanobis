import numpy as np
from utils.utils import split_arrays_ano, NumpyEncoder
import pandas as pd
import json
import argparse


def load_kpi(path="kpi.pkl", seq_len=120, stride=120, test_size=0.5, random_state=42):
    datas = pd.read_pickle(path)
    outputs = {}
    for key in datas["all_train_data"].keys():
        sub_output = {}
        X_large = datas["all_train_data"][key]
        y_large = datas["all_train_labels"][key]
        (X_tr, y_tr,
         X_te, y_te) = split_arrays_ano(X_large, y_large, seq_len, stride,
                                        test_size, random_state, split=False)
        sub_output["X_train"] = X_tr
        sub_output["y_train"] = y_tr
        sub_output["X_test"] = X_te
        sub_output["y_test"] = y_te
        outputs[key] = sub_output
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='input and output path.')

    parser.add_argument('--input_path', type=str, default="./kpi.pkl", help='path to the pkl file (input)')
    parser.add_argument('--output_path', type=str, default="./kpi.json", help='path to the json file (output)')

    args = parser.parse_args()

    data = load_kpi(args.input_path)

    with open(args.output_path, "w") as file:
        json.dump(data, file, cls=NumpyEncoder, indent=4)
