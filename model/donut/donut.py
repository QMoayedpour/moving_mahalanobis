import math
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from utils.dataloaders import SimpleDataset
from tqdm import tqdm
from utils.metrics import all_metrics
from utils.utils import (get_modif_score,
                         split_arrays_ano,
                         create_windows,
                         score_windows,
                         reconstruct)


class VAE(nn.Module):
    def __init__(self,
                 seq_len=120,
                 number_of_neural_per_layer=100,
                 latent_dim=8,
                 num_l_samples=64,
                 activation_function=nn.ReLU(),
                 device="cpu"):
        super(VAE, self).__init__()

        self.n_l_samples = num_l_samples
        self.seq_len = seq_len
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, number_of_neural_per_layer),
            activation_function,
            nn.Linear(number_of_neural_per_layer, number_of_neural_per_layer),
            activation_function,
        )

        self.en_miu = nn.Linear(number_of_neural_per_layer, latent_dim)
        self.en_std = nn.Sequential(
            nn.Linear(number_of_neural_per_layer, latent_dim),
            nn.Softplus()
        )
        self.epsilon = 0.0001
        self.device = device

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, number_of_neural_per_layer),
            activation_function,
            nn.Linear(number_of_neural_per_layer, number_of_neural_per_layer),
            activation_function,
        )

        self.de_miu = nn.Linear(number_of_neural_per_layer, seq_len)
        self.de_std = nn.Sequential(
            nn.Linear(number_of_neural_per_layer, seq_len),
            nn.Softplus()
        )

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        """
        :param x:
        :param n_sample: z的采样次数
        :return:
        """
        if self.training:

            encoder_out = self.encoder(x)
            z_miu = self.en_miu(encoder_out)
            z_std = self.en_std(encoder_out) + self.epsilon
            z = z_miu + z_std * torch.randn(z_miu.shape[0], z_miu.shape[1]).to(self.device)

            decoder_out = self.decoder(z)
            x_bar_miu = self.de_miu(decoder_out)
            x_bar_std = self.de_std(decoder_out) + self.epsilon
            return z, x_bar_miu, x_bar_std, z_miu, z_std

        else:
            encoder_out = self.encoder(x)
            z_miu = self.en_miu(encoder_out)
            z_std = self.en_std(encoder_out) + self.epsilon
            batch_size = z_miu.shape[0]

            z_miu = z_miu.repeat(self.n_l_samples, 1).view(self.n_l_samples, batch_size, -1)
            z_std = z_std.repeat(self.n_l_samples, 1).view(self.n_l_samples, batch_size, -1)
            z = torch.normal(mean=z_miu, std=z_std)

            decoder_out = self.decoder(z)
            x_bar_miu = self.de_miu(decoder_out)
            x_bar_std = self.de_std(decoder_out) + self.epsilon
            return z, x_bar_miu, x_bar_std, z_miu, z_std


class Donut:
    def __init__(self,
                 X=None,
                 y=None,
                 dataloader=None,
                 lr=0.001,
                 weight_decay=0.001,
                 seq_len=120,
                 latent_dim=3,
                 number_of_neural_per_layer=120,
                 num_l_samples=64,
                 activation_function=nn.ReLU(),
                 batch_size=256,
                 step_size=30,
                 gamma=0.1,
                 n_epoch=30,
                 device="cpu",
                 verbose=True,
                 model_name="Donut",
                 windows=False,
                 eval=False,
                 test_size=0.8, **kwargs):
        self._batch_size = batch_size
        self._opti_step_size = step_size
        self._epoch = n_epoch
        self._opti_gamma = gamma
        self.seq_len = seq_len
        self._vae = VAE(seq_len=seq_len,
                        latent_dim=latent_dim,
                        number_of_neural_per_layer=number_of_neural_per_layer,
                        num_l_samples=num_l_samples,
                        activation_function=activation_function,
                        device=device
                        ).to(device)
        self.device = device
        self.windows = windows
        self.verbose = verbose
        self.optimizer = Adam(self._vae.parameters(), lr=lr, weight_decay=weight_decay)
        self.X = X
        self.y = y
        self.test_size = test_size
        self.split = True
        self.model_name = model_name

    def set_params(self, **params):
        for param_name, param_value in params.items():
            setattr(self, param_name, param_value)
        self.__init__(self, **params)

    def fit(self):
        if self.X is None or self.y is None:
            raise ValueError("Please assign a value to X and y (use set_params())")

        X = np.array(self.X).flatten()
        y = np.array(self.y).flatten()

        (X_train, y_train,
         X_test, y_test) = split_arrays_ano(X, y, seq_len=self.seq_len,
                                            stride=1, split=self.split, test_size=self.test_size)
        if self.windows:

            X_test = create_windows(reconstruct(X_test, seq_len=self.seq_len, stride=1),
                                    seq_len=self.seq_len,
                                    stride=self.seq_len)
            y_test = create_windows(reconstruct(y_test, seq_len=self.seq_len, stride=1),
                                    seq_len=self.seq_len,
                                    stride=self.seq_len)

        self.train_and_eval(X_train, y_train)

        self.score, _, __, ___ = self.predict(X_test, y_test)
        if self.windows:

            self.labels = y_test.flatten()

            labels, score = score_windows(self.labels, self.score, seq_len=self.seq_len)

        else:
            self.labels = y_test[:, -1]
            labels, score = self.labels, self.score

        out = all_metrics(labels, score, n_data=X_test.shape[0],
                          model=self.model_name)

        return out

    def m_elbo_loss(self, train_x, train_y, z, x_bar_miu, x_bar_std, z_miu, z_std, device="cpu"):
        """
        L=1
        :param train_x: batch_size * win
        :param train_y: batch_size * 1,
        :param z: batch_size * latent_size
        :param x_bar_miu: batch_size * win
        :param x_bar_std: batch_size * win
        :param z_miu: batch_size * latent_size
        :param z_std: batch_size * latent_size
        :param z_prior_mean: int
        :param z_prior_std: int
        :return: loss
        """

        z_prior_mean = torch.zeros(size=z_miu.shape).to(device)
        z_prior_std = torch.ones(size=z_miu.shape).to(device)

        log_p_x_given_z = - torch.log(math.sqrt(2 * math.pi) * x_bar_std) - ((train_x - x_bar_miu) ** 2) / (
                2 * x_bar_std ** 2)

        log_p_z = - torch.log(math.sqrt(2 * math.pi) * z_prior_std) - ((z - z_prior_mean) ** 2) / (
                    2 * z_prior_std ** 2)

        log_q_z_given_x = - torch.log(math.sqrt(2 * math.pi) * z_std) - ((z - z_miu) ** 2) / (2 * z_std ** 2)

        normal = 1 - train_y  # batch_size * win
        log_p_x_given_z = normal * log_p_x_given_z  # batch_size * win

        beta = torch.sum(normal, dim=1) / normal.shape[1]  # size = batch_size

        m_elbo = torch.sum(log_p_x_given_z, dim=1) + beta * torch.sum(log_p_z, dim=1) - torch.sum(log_q_z_given_x,
                                                                                                  dim=1)
        m_elbo = torch.mean(m_elbo) * (-1)
        return m_elbo

    def train_and_eval(self, x, y, valid_x=None, valid_y=None):
        """
        :param x:training set, (n_samples, sliding_window)
        :param y:testing set,  (n_samples, sliding_window)
        :param valid_x:  (n_samples, sliding_window)
        :param valid_y:  (n_samples, sliding_window)
        Parameters
        ----------
        x
        y
        valid_x
        valid_y

        Returns
        -------

        """
        self._vae.train(mode=True)
        train_dataset = SimpleDataset(x, y, seq_len=x.shape[1], device=self.device)
        train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True,
                                                 num_workers=0)
        valid_iter = None
        loss = None

        if valid_x is not None:
            valid_dataset = SimpleDataset(valid_x, valid_y, seq_len=self.seq_len,
                                          device=self.device)
            valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=self._batch_size, shuffle=False,
                                                     num_workers=0)

        optimizer = self.optimizer
        lr_scheduler = StepLR(optimizer, step_size=self._opti_step_size, gamma=self._opti_gamma)
        epoch_bar = tqdm(range(self._epoch), desc="Training", unit="epoch") if self.verbose else range(self._epoch)

        for epoch in epoch_bar:
            for train_x, train_y in train_iter:
                optimizer.zero_grad()
                z, x_bar_miu, x_bar_std, z_miu, z_std = self._vae.forward(train_x)
                loss = self.m_elbo_loss(train_x, train_y, z, x_bar_miu, x_bar_std, z_miu, z_std, self.device)
                loss.backward()
                optimizer.step()
            lr_scheduler.step()

            if epoch % 10 == 0 and valid_x is not None:
                with torch.no_grad():
                    for v_x, v_y in valid_iter:
                        z, x_bar_miu, x_bar_std, z_miu, z_std = self._vae.forward(v_x)
                        v_l = self.m_elbo_loss(v_x, v_y, z, x_bar_miu, x_bar_std, z_miu, z_std, self.device)
                    if self.verbose:
                        epoch_bar.set_postfix(loss=f"{v_l.item():.4f}")

    def predict(self, test_x, test_y):
        """
        Predict
        Parameters
        ----------
        test_x: Testing set of shape (n_samples,sliding_windows)
        test_y: Testing labels of shape (n_samples,sliding_windows)
        Returns
        -------
        ret_scores,ret_modified_score, ret_x_bar_mean, ret_x_bar_std
        """
        # Sets the module in evaluation mode so we can predict.
        # It is actually calling the function: self.train(False)
        self._vae.eval()  # equal to self.train(False)

        test_dataset = SimpleDataset(test_x, test_y, seq_len=self.seq_len,
                                     device=self.device)
        test_iter = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
        ret_scores = []
        ret_x_bar_mean = []
        ret_x_bar_std = []
        with torch.no_grad():
            for batch_x, batch_y in test_iter:
                z, x_bar_mean, x_bar_std, z_miu, z_std = self._vae(batch_x)
                log_p_x_given_z = torch.log(math.sqrt(2 * math.pi) * x_bar_std) + \
                                   ((batch_x - x_bar_mean) ** 2) / (2 * x_bar_std ** 2)
                if self.windows:

                    anomaly_score = torch.mean(log_p_x_given_z, dim=0)

                    #anomaly_score, _ = torch.max(anomaly_score, dim=1)

                else:
                    anomaly_score = torch.mean(log_p_x_given_z[:, :, -1], dim=0)

                ret_scores.append(anomaly_score)
                ret_x_bar_mean.append(torch.mean(x_bar_mean[:, :, -1], dim=0))
                ret_x_bar_std.append(torch.mean(x_bar_std[:, :, -1], dim=0))

            ret_scores = torch.cat(ret_scores)
            ret_x_bar_mean = torch.cat(ret_x_bar_mean)
            ret_x_bar_std = torch.cat(ret_x_bar_std)
            assert len(ret_scores) == len(test_dataset)
            if not self.windows:
                ret_modified_score = get_modif_score(test_y[:, -1],
                                                     ret_scores.cpu().detach().numpy())
            else:
                ret_modified_score = ret_scores.cpu().detach().numpy()
            return ret_scores.cpu().detach().numpy().flatten(), ret_modified_score, ret_x_bar_mean, ret_x_bar_std