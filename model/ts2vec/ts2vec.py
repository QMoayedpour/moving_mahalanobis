import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
from tqdm import trange
from .layers import TSEncoder, hierarchical_contrastive_loss
from torch.utils.data import TensorDataset, DataLoader
from .utils import (take_per_row,
                    split_with_nan,
                    centerize_vary_length_series,
                    torch_pad_nan,
                    convert_pandas_to_data_in)


class TS2Vec_Learner:
    def __init__(
        self,
        train_data,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cuda',
        batch_size=16,
        max_train_length=None,
        temporal_unit=0,
        after_iter_callback=None,
        after_epoch_callback=None
    ):
        super().__init__()

        assert train_data.ndim == 3, "Not a valid format, make sure datashape are (B, T, D)"
        self.train_data = train_data  # train_data must be pre-process
        self.device = device
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit

        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims,
                              hidden_dims=hidden_dims, depth=depth).to(self.device)
        self.net = AveragedModel(self._net)
        self.net.update_parameters(self._net)

        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback

        self.n_epochs = 0
        self.n_iters = 0
        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)

        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]

        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=min(self.batch_size, len(train_dataset)),
                                       shuffle=True, drop_last=True)

    def fit(self, n_epochs=200, verbose=False, lr=1e-4):

        optimizer = torch.optim.AdamW(self._net.parameters(), lr=lr)

        loss_log = []

        for epoch in trange(n_epochs):

            cum_loss = 0
            n_epoch_iters = 0
            for batch in self.train_loader:
                x = batch[0]
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset: window_offset + self.max_train_length]
                x = x.to(self.device)

                ts_l = x.size(1)
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1,
                                                size=x.size(0))

                optimizer.zero_grad()

                out1 = self._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
                out1 = out1[:, -crop_l:]

                out2 = self._net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
                out2 = out2[:, :crop_l]

                loss = hierarchical_contrastive_loss(
                    out1,
                    out2,
                    temporal_unit=self.temporal_unit
                )

                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)

                cum_loss += loss.item()
                n_epoch_iters += 1

                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())

            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1

            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)

        return loss_log

    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        out = self.net(x.to(self.device, non_blocking=True), mask)
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=out.size(1),
            ).transpose(1, 2)

        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=encoding_window,
                stride=1,
                padding=encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]

        else:
            if slicing is not None:
                out = out[:, slicing]

        return out.cpu()

    def encode(self, data, mask=None, encoding_window=None, causal=False, sliding_length=None,
               sliding_padding=0):
        assert self.net is not None, 'please train with .fit()'

        if isinstance(data, pd.DataFrame):
            data = convert_pandas_to_data_in(data)

        assert data.ndim == 3
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float)
        x = data.to(self.device)
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()

        with torch.no_grad():
            output = []
            if sliding_length is not None:
                reprs = []
                for i in range(0, ts_l, sliding_length):
                    l = i - sliding_padding
                    r = i + sliding_length + (sliding_padding if not causal else 0)
                    x_sliding = torch_pad_nan(
                        x[:, max(l, 0) : min(r, ts_l)],
                        left=-l if l<0 else 0,
                        right=r-ts_l if r>ts_l else 0,
                        dim=1
                    )
                    out = self._eval_with_pooling(
                        x_sliding,
                        mask,
                        slicing=slice(sliding_padding, sliding_padding+sliding_length),
                        encoding_window=encoding_window
                    )
                    reprs.append(out)

                out = torch.cat(reprs, dim=1)
                if encoding_window == 'full_series':
                    out = F.max_pool1d(
                        out.transpose(1, 2).contiguous(),
                        kernel_size=out.size(1),
                    ).squeeze(1)
            else:
                out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                if encoding_window == 'full_series':
                    out = out.squeeze(1)

            output.append(out)

            output = torch.cat(output, dim=0)

        self.net.train(org_training)
        return output.numpy()
