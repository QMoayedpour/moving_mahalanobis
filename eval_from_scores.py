import argparse
import numpy as np
import json
from tqdm import tqdm
from utils.metrics import all_metrics
from utils.utils import score_windows, push_json
from print_results import print_dic_results


def point_to_window(labels, scores, seq_len=120):

    labels = np.array(labels)
    scores = np.array(scores)

    labels, scores = score_windows(labels, scores, seq_len=seq_len)
    return all_metrics(labels, scores, verbose=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Model Trainer.')

    parser.add_argument('--path', type=str, default="./scores_lagllama_KPI.json", help='path to score dict (json)')
    parser.add_argument('--seq_len', type=int, default=120, help='sequence length for TS windows')
    parser.add_argument('--save_new_res', type=str, default="", help='Path to save the new results')

    args = parser.parse_args()

    with open(args.path, 'r') as fichier:
        dictionnaire = json.load(fichier)

    all_results = {}

    for key in tqdm(dictionnaire.keys()):
        (labels, scores) = (dictionnaire[key]["labels"], dictionnaire[key]["scores"])
        all_results[key] = point_to_window(labels, scores, args.seq_len)

    if args.save_new_res != "":
        push_json(args.save_new_res, all_results)

    print(all_results['05f10d3a-239c-3bef-9bdc-a2feeb0037aa_t1'])
    print_dic_results(all_results)
