from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset


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