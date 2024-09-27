import argparse
import json
import pandas as pd
import warnings
import math
warnings.filterwarnings('ignore')


def print_dic_results(path, dec=3):

    if isinstance(path, str):
        with open(path, "r") as file:
            data = json.load(file)
        results = compute_mean_dic(data)

    elif isinstance(path, dict):
        results = compute_mean_dic(path)

    else:
        raise ValueError("Input must be a path to a json file or a dictionnary")

    for key in results.keys():
        value = results[key]
        if isinstance(value, float) and not pd.isna(value):
            print(f"{key}: {round(value, dec)}")


def compute_mean_dic(results):
    metrics_sum = {}
    metrics_count = {}

    for model_name, values in results.items():
        for metric, value in values.items():
            if isinstance(value, (int, float)) and not math.isnan(value):
                if metric not in metrics_sum:
                    metrics_sum[metric] = 0
                    metrics_count[metric] = 0
                metrics_sum[metric] += value
                metrics_count[metric] += 1

    average_metrics = {metric: (metrics_sum[metric] / metrics_count[metric]) 
                       for metric in metrics_sum if metrics_count[metric] > 0}

    return average_metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Model Trainer.')

    parser.add_argument('--path', type=str, default="./result_anomaly/lagllama_mahala\
/lagllama_padim_kpi.json", help='path to the json file')
    parser.add_argument('--n_decimals', type=int, default=3, help='number of decimals to print')
    args = parser.parse_args()

    print_dic_results(args.path, args.n_decimals)
