import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm
from utils.utils import NumpyEncoder
import json
import os
import glob


def extract_dataset_name(text):
    start_index = text.rfind('/')
    end_index = text.rfind('.csv')
    if start_index != -1 and end_index != -1 and start_index < end_index:
        return text[start_index + 1:end_index]
    return None


def pandas_to_json_nab(data_dir="./"):
    datasets = glob.glob(os.path.join((data_dir), "*.csv"))

    data = {}
    for path in tqdm(datasets):
        key = extract_dataset_name(path)

        df = pd.read_csv(path)

        X = df.value.to_numpy()

        y = df.label.to_numpy()

        data[key] = {"X": X, "y": y}

    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='input and output path.')

    parser.add_argument('--input_folder', type=str, default="./", help='path to the folder (where csv data are) (input)')
    parser.add_argument('--output_path', type=str, default="./kpi.json", help='path to the json file (output)')

    args = parser.parse_args()

    data = pandas_to_json_nab(args.input_folder)

    with open(args.output_path, "w") as file:
        json.dump(data, file, cls=NumpyEncoder, indent=4)
