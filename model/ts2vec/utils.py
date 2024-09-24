import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.utils import create_windows
from torch.utils.data import Dataset


def take_per_row(A, indx, num_elem):
    all_indx = indx[:, None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]


def torch_pad_nan(arr, left=0, right=0, dim=0):
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
    return arr


def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]


def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size//2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)


def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs


def generate_continuous_mask(B, T, n=5, length=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)

    if isinstance(length, float):
        length = int(length * T)
    length = max(length, 1)

    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-length + 1)
            res[i, t:t+length] = False
    return res


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.isocalendar().week.to_numpy(),
    ], axis=1).astype(float)


def get_data(dataframe="Twitter_volume_UPS", split_ratios=[0.6, 0.2, 0.2], path="./data_anomaly/"):
    assert sum(split_ratios) == 1, "Les proportions doivent avoir une somme égale à 1"
    path_to_data = path + dataframe + ".csv"
    df = pd.read_csv(path_to_data, index_col="timestamp", parse_dates=True)
    time_ft = _get_time_features(df.index)
    n_covariate_cols = time_ft.shape[-1]

    serie = df.value.to_numpy()
    n = serie.shape[0]
    train_size = int(n * split_ratios[0])
    val_size = int(n * split_ratios[1])
    test_size = n - train_size - val_size

    train_slice = slice(None, train_size)
    valid_slice = slice(train_size, test_size)
    test_slice = slice(test_size, n)

    scaler = StandardScaler().fit(serie[train_slice].reshape(-1, 1))
    serie = scaler.transform(serie.reshape(-1, 1))
    serie = np.expand_dims(serie, 0)
    if n_covariate_cols > 0:
        time_scaled = MinMaxScaler().fit(time_ft[train_slice])
        time_ft = np.expand_dims(time_scaled.transform(time_ft), 0)
        data = np.concatenate([np.repeat(time_ft, serie.shape[0], axis=0), serie], axis=-1)
    pred_lens = [24, 48, 96, 288, 672]
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols


def convert_pandas_to_data_in(df):
    assert "value" in df.columns, "not a valid format, make sure [value] is a column"
    cond = pd.api.types.is_datetime64_any_dtype(df.index)
    assert cond, "index not in datetime"
    time_ft = _get_time_features(df.index)
    n_covariate_cols = time_ft.shape[-1]

    serie = df.value.to_numpy()
    scaler = StandardScaler().fit(serie.reshape(-1, 1))
    serie = scaler.transform(serie.reshape(-1, 1))
    serie = np.expand_dims(serie, 0)
    if n_covariate_cols > 0:
        time_scaled = MinMaxScaler().fit(time_ft)
        time_ft = np.expand_dims(time_scaled.transform(time_ft), 0)
        data = np.concatenate([np.repeat(time_ft, serie.shape[0], axis=0), serie], axis=-1)

    return data

def preprocess_x(etth2_path):
    data_pretrain = pd.read_csv(etth2_path).rename({'OT': 'value'}, axis=1)

    X = data_pretrain.value.to_numpy()
    X = X.reshape(-1, 1)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X).reshape(1,X.shape[0], 1)
    X = X_scaled.reshape(1,X.shape[0], 1)
    return X


def create_ts2vec_dataset(x, model, seq_len=32, stride=32, device="cpu"):
    dataset = WindowDataset(x, seq_len, stride, device)
    return TS2VecDataset(dataset, model, device)


class WindowDataset(Dataset):
    def __init__(self, x, seq_len=32, stride=32, device='cpu'):
        self.seq_len = seq_len 
        self.stride = stride
        self.x = create_windows(x, seq_len=self.seq_len, stride=self.stride)

        self.device = device

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = torch.tensor(self.x[idx], dtype=torch.float32).to(self.device)
        return sample


class TS2VecDataset(Dataset):
    def __init__(self, dataset, model, device="cuda"):
        self.dataset = dataset
        self.model = model
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx].unsqueeze(0).unsqueeze(2)

        return self.model.encode(item)
