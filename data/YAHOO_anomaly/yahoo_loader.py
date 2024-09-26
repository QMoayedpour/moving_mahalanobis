import os
import pandas as pd
import numpy as np
import argparse
import json
from utils.utils import NumpyEncoder


def get_yahoo_index(directory_path):
    indices = set()
    for filename in os.listdir(directory_path):
        if filename.endswith('.out'):
            index = filename.split('_data.')[0]
            indices.add(index)

    return list(indices)


def get_yahoo_data(index, directory_path):

    df = pd.read_csv(directory_path + f"/{index}_data.out", header=None)

    X = df[0].to_numpy()
    y = df[1].to_numpy()

    return {"X":X, "y":y}


def extract_yahoo_data(directory_path):
    dataset = get_yahoo_index(directory_path)

    data = {}

    for key in dataset:
        data[key] = get_yahoo_data(key, directory_path)

    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='input and output path.')

    parser.add_argument('--input_folder', type=str, default="./", help='path to the folder (where csv data are) (input)')
    parser.add_argument('--output_path', type=str, default="./data/YAHOO_anomaly/YAHOO.json", help='path to the json file (output)')

    args = parser.parse_args()

    data = extract_yahoo_data(args.input_folder)

    with open(args.output_path, "w") as file:
        json.dump(data, file, cls=NumpyEncoder, indent=4)