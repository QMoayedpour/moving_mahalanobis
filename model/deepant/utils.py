import pandas as pd
import numpy as np
import torch
import logging
from tqdm import tqdm, trange
from utils.dataloaders import get_deepant_dataloaders
from utils.metrics import all_metrics
from .deepant import DeepAnt, AnomalyDetector
from .predictor import DeepAntPredict
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from utils.utils import create_windows, score_windows


class DeepAntLearner:
    def __init__(self, X=None, y=None, seq_len=32, dataloader=None, windows=False,
                 n_epochs=150, model_name="DeepAnt", batch_size=32, device="cuda", **kwargs):

        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.dataloader = dataloader
        self.windows = windows
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.model_name = model_name
        self.device = device
        self.lr = 1e-5
        for param_name, param_value in kwargs.items():
            setattr(self, param_name, param_value)

    def fit(self):
        
        self.X = np.array(self.X)
        train_loader = get_deepant_dataloaders(self.X, self.seq_len, self.batch_size)

        model = DeepAnt(self.seq_len, 1, self.seq_len)

        self.predictor = DeepAntPredict(model, lr=self.lr)

        self.predictor.train(train_loader, self.n_epochs)


        _, errors = self.predictor.predict(self.X, self.seq_len)

        self.score = np.concatenate([np.array([0]*self.seq_len), errors])

        self.labels = np.array(self.y)

        self.labels[:self.seq_len] = 0

        if self.windows:
            labels, score = score_windows(self.labels, self.score, seq_len=self.seq_len)

        else:
            labels, score = self.labels, self.score

        out = all_metrics(labels, score, model=self.model_name)

        return out
