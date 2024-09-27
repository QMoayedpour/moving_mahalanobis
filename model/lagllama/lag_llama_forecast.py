import numpy as np
from tqdm import trange, tqdm
import pandas as pd
import glob
import os
import json
from utils.dataloaders import LForecastDataset
from utils.metrics import all_metrics
from utils.utils import create_windows
import torch
from model.lagllama.gluon.estimator import LagLlamaEstimator
from gluonts.dataset.common import ListDataset
from torch.utils.data import DataLoader


class LagLLamaForecastAno:
    def __init__(self, X=None, y=None, seq_len=32, batch_size=64, device="cuda",
                 model_path="./moving_mahalanobis/model/lagllama/lag-llama.ckpt",
                 model_name="LagLlamaForecast",
                 verbose=False, windows=False,
                 **kwargs):
        self.X = X
        self.y = y
        self.device = torch.device(device)
        self.seq_len = 32
        self.batch_size = batch_size
        self.model_path = model_path
        self.scores = []
        self.labels = []
        self.verbose = verbose
        self.windows = windows
        self.model_name = model_name
        for param_name, param_value in kwargs.items():
            setattr(self, param_name, param_value)       

    def fit(self):
        ckpt = torch.load(self.model_path, map_location=self.device)
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

        rope_scaling_arguments = {
            "type": "linear",
            "factor": max(1.0, (self.seq_len + 1) / estimator_args["context_length"]),
        }

        estimator = LagLlamaEstimator(
            ckpt_path=self.model_path,
            prediction_length=1,
            context_length=self.seq_len,
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            scaling=estimator_args["scaling"],
            time_feat=estimator_args["time_feat"],
            rope_scaling=None,
            batch_size=self.batch_size,  # Utiliser un batch size plus grand
            num_parallel_samples=10,
            device=self.device,
        )

        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        predictor = estimator.create_predictor(transformation, lightning_module)

        X = np.array(self.X).flatten()
        y = np.array(self.y).flatten()

        dataset = LForecastDataset(X, self.seq_len)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        squared_errors = []

        for batch in tqdm(data_loader) if self.verbose else data_loader:
            batch_dataset = ListDataset(
                [{"start": pd.Timestamp("2000-01-01"), "target": seq.numpy()} for seq in batch],
                freq='D'
            )

            predictions = list(predictor.predict(batch_dataset))

            for i, prediction in enumerate(predictions):
                predicted_value = prediction.mean[-1]
                actual_value = X[len(squared_errors) + self.seq_len]
                squared_error = (predicted_value - actual_value) ** 2
                squared_errors.append(squared_error)

        errors, self.labels = squared_errors, y
        errors = [0] * self.seq_len + errors
        self.score = np.array(errors)

        if self.windows:
            preds = create_windows(errors, seq_len=self.seq_len, stride=self.seq_len)
            preds = np.max(preds, axis=1)

            gt_list = create_windows(self.labels, seq_len=self.seq_len, stride=self.seq_len)
            gt_list = np.max(gt_list, axis=1)

        else:
            preds = self.score
            gt_list = self.labels

        out = all_metrics(gt_list, preds, model=self.model_name)

        return out
