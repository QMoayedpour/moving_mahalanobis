import torch
import torch.nn as nn
import pytorch_lightning as pl


class DeepAnt(nn.Module):
    def __init__(self, seq_len, p_w, window_size):
        super().__init__()
        
        self.convblock1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding='valid'),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)
        )

        self.convblock2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding='valid'),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)
        )

        self.flatten = nn.Flatten()
        self.conv_output_dim = int(self.calculate_conv_output_dim(window_size))
        self.denseblock = nn.Sequential(
            #nn.Linear(32, 40),
            nn.Linear(self.conv_output_dim, 40),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
        )
        self.out = nn.Linear(40, p_w)

    def calculate_conv_output_dim(self, window_size):
        output_dim = window_size
        output_dim = output_dim - 2
        output_dim = output_dim // 2

        output_dim = output_dim - 2
        output_dim = output_dim // 2

        return output_dim * 32

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.flatten(x)
        x = self.denseblock(x)
        x = self.out(x)
        return x


class AnomalyDetector(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.L1Loss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        return y_pred, torch.linalg.norm(y_pred-y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)
