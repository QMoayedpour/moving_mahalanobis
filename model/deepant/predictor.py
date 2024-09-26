import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from utils.utils import create_windows
from utils.dataloaders import get_deepant_dataloaders
from model.deepant.deepant import DeepAnt
from tqdm import trange


class DeepAntPredict:
    def __init__(self, model, lr=1e-5, device=None):
        self.model = model
        self.criterion = nn.L1Loss()
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def train(self, train_loader, n_epochs):
        self.model.train()
        for epoch in trange(n_epochs):
            epoch_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

    def predict(self, X, seq_len):
        self.model.eval()
        self.model = self.model.to(self.device)

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X.reshape(-1, 1))

        windows = create_windows(X_scaled, seq_len=seq_len, stride=1)

        predictions = []
        errors = []

        with torch.no_grad():
            for i, window in enumerate(windows):
                window_tensor = torch.tensor(window, dtype=torch.float).unsqueeze(0).permute(0, 2, 1).to(self.device)

                prediction = self.model(window_tensor)

                true_value = torch.tensor(X_scaled[i + seq_len])

                error = torch.abs(prediction.item() - true_value).item()

                predictions.append(prediction.item())
                errors.append(error)
        predictions = [float(x) for x in predictions]
        errors = [float(x) for x in errors]
        return np.array(predictions), np.array(errors)
