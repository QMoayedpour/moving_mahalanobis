import pandas as pd
import numpy as np
from tqdm import tqdm
from gluonts.dataset.common import ListDataset
from torch.utils.data import DataLoader, Dataset
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class ForecastDataset(Dataset):
    def __init__(self, X, context_length):
        self.X = X
        self.context_length = context_length

    def __len__(self):
        return len(self.X) - self.context_length

    def __getitem__(self, idx):
        return self.X[idx: idx + self.context_length]


def normalise(array):
    if isinstance(array, list):
        array = np.array(array)
    return (array - array.min()) / (array.max() - array.min())


def moirai_ano(model, serie, y=None, context_length=32, batch_size=32,):

    if not isinstance(y, np.ndarray):
        y = np.zeros(len(serie))
    serie = normalise(serie)
    predictor = model.create_predictor(batch_size=batch_size)

    dataset = ForecastDataset(serie, context_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    squared_errors = []
    labels = []
    for batch in tqdm(data_loader):
        batch_dataset = ListDataset(
            [{"start": pd.Timestamp("2000-01-01"), "target": seq.numpy()} for seq in batch],
            freq='D'
        )

        predictions = list(predictor.predict(batch_dataset))

        for i, prediction in enumerate(predictions):
            predicted_value = prediction.mean[-1]
            actual_value = serie[len(squared_errors) + context_length]
            label = y[len(squared_errors) + context_length]
            labels.append(label)
            squared_error = (predicted_value - actual_value) ** 2
            squared_errors.append(squared_error)

    return {"scores": squared_errors, "labels": labels}


def push_json(output_path, dic):
    path = output_path
    with open(path, 'w') as json_file:
        json.dump(dic, json_file, indent=4)
    print(f"results saved @ {path}")


def create_windows(array, seq_len=120, stride=120):
    array_list = []
    for i in range(0, len(array) - seq_len, stride):
        array_list.append(array[i:i+seq_len])
    return np.array(array_list)


def score_windows(labels, score, seq_len=120):
    labels = create_windows(labels, seq_len=seq_len, stride=seq_len)
    score = create_windows(score, seq_len=seq_len, stride=seq_len)

    return np.max(labels, axis=1), np.max(score, axis=1)


def split_arrays_ano(X_large, y_large, seq_len=120, stride=120, test_size=0.5, random_state=42,
                     split=False):
    """take arrays to split subarrays for test and train. Train contains 0 sub array with anomaly

    Args:
        X_large (np.array): Value 1d array
        y_large (np.array): Labels 1d array
        seq_len (int): Size of sub arrays
        stride (int): stride

    Returns:
        datasets: X_train, y_train, X_test, y_test
    """
    X = []
    y = []
    y_mask = []
    for i in range(0, len(X_large) - seq_len, stride):
        X.append(X_large[i:i+seq_len])
        y.append(y_large[i:i+seq_len])
        y_mask.append(int(y_large[i:i+seq_len].sum()))
    X = np.stack(X, axis=0)
    y = np.stack(y, axis=0)
    y_mask = np.stack(y_mask, axis=0)

    scaler = MinMaxScaler()
    X_shape = X.shape
    X = scaler.fit_transform(X.reshape(-1, X_shape[-1])).reshape(X_shape)

    X_train, X_test, y_train, y_test, y_train_mask, y_test_mask = train_test_split(
        X, y, y_mask, test_size=test_size, random_state=random_state, shuffle=False
    )
    if split:
        class_to_keep, class_to_remove = 0,  1

        mask_train = y_train_mask == class_to_keep
        mask_train_remove = y_train_mask >= class_to_remove

        X_train_healthy = X_train[mask_train]
        y_train_healthy = y_train[mask_train]

        X_train_anomaly = X_train[mask_train_remove]
        y_train_anomaly = y_train[mask_train_remove]
        X_test = np.concatenate((X_train_anomaly, X_test), axis=0)
        y_test = np.concatenate((y_train_anomaly, y_test), axis=0)
    else:
        X_train_healthy = X_train
        y_train_healthy = y_train
    return (X_train_healthy, y_train_healthy, X_test, y_test)
