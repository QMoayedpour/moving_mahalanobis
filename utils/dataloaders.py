from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.utils import create_windows
import pytorch_lightning as pl


def get_deepant_dataloaders(X, seq_len, batch_size):
    dataset = DeepAntDataset(X, seq_len)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              num_workers=10, pin_memory=True)
    return train_loader


class SimpleDataset(Dataset):
    def __init__(self, X, y, seq_len=32, device="cpu", stride=1):
        self.scaler = MinMaxScaler()
        self.X = self.scaler.fit_transform(X)
        #self.X = X
        self.y = y
        self.seq_len = seq_len
        if stride > 1:
            self.seq_len = 0
        self.device = device

    def __len__(self):
        return len(self.X)  # - self.seq_len

    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx], dtype=torch.float).to(self.device),
                torch.tensor(self.y[idx], dtype=torch.int).to(self.device))


class DataModule(pl.LightningDataModule):
    def __init__(self, X, y, seq_len=120, batch_size=32):
        super().__init__()
        self.X = np.array(X)
        self.y = y
        self.seq_len = seq_len
        self.bs = batch_size

    def setup(self, stage=None):
        self.dataset = DeepAntDataset(self.X, self.y, self.seq_len)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.bs, num_workers=10,
                          pin_memory=True, shuffle=True)

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=1, num_workers=10,
                          pin_memory=True, shuffle=False)


class DeepAntDataset(Dataset):
    def __init__(self, X, seq_len):
        self.X = np.array(X)
        self.seq_len = seq_len
        self.sequence, self.labels = self.create_sequence(self.X, seq_len)

    def create_sequence(self, X, seq_len):
        self.sc = MinMaxScaler()
        self.ts = self.sc.fit_transform(X.reshape(-1, 1))

        ts = create_windows(self.ts, seq_len=seq_len, stride=1)[:-1,:]
        label = create_windows(self.ts[1:], seq_len=seq_len, stride=1)[:, -1]
        return ts, label

    def __len__(self):
        return self.sequence.shape[0]

    def __getitem__(self, idx):
        return (torch.tensor(self.sequence[idx], dtype=torch.float).permute(1, 0),
                torch.tensor(self.labels[idx], dtype=torch.float))
