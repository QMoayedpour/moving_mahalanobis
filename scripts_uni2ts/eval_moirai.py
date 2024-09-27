import pandas as pd
import numpy as np
from tqdm import tqdm
from metrics import all_metrics
import argparse
import json
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from anom_utils import moirai_ano, push_json, score_windows


def eval_uni2ts(seq_len=64, seq_len_windows=32, batch_size=64, dataset_path="./NAB.json",
                dataset_name="NAB", save_scores=False):

    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-small"),
        prediction_length=1,
        context_length=seq_len,
        patch_size="auto",
        num_samples=10,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )

    with open(dataset_path, "r") as file:
        datasets = json.load(file)

    results = {}
    results_windows = {}

    for key in tqdm(datasets.keys()):

        X = np.array(datasets[key]["X"]).flatten()
        y = np.array(datasets[key]["y"]).flatten()    

        res = moirai_ano(model, X, y, seq_len, batch_size)

        results[key] = all_metrics(res["labels"], res["scores"], model="MOIRAI")

        labels_windows, scores_windows = score_windows(res["labels"], res["scores"],
                                                       seq_len=seq_len_windows)

        results_windows[key] = all_metrics(labels_windows, scores_windows, model="MOIRAI")

        res["labels"] = [float(x) for x in res["labels"]]
        res["scores"] = [float(x) for x in res["scores"]]
        push_json(f"./results_{dataset_name}.json", results)
        push_json(f"./result_windows_{dataset_name}", results_windows)
        if save_scores:
            push_json(f"./scores_{dataset_name}.json", res)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Model Trainer.')
    parser.add_argument('--path', type=str, default="./NAB.json", help='path to the json file (data)')
    parser.add_argument('--dataset_name', type=str, default="NAB", help='name of the dataset')
    parser.add_argument('--seq_len', type=int, default=64, help='context length for the model')
    parser.add_argument('--seq_len_windows', type=int, default=1, help='seq_len for windows\
         evaluation (1 = point-wise)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for the model')
    parser.add_argument('--save_scores', type=bool, default=True, help='To save scores or not')
    args = parser.parse_args()

    eval_uni2ts(args.seq_len, args.seq_len_windows, args.batch_size,
                args.path, args.dataset_name, args.save_scores)
