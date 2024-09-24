import torch
import pandas as pd
from datetime import datetime
import numpy as np
from model.lagllama.gluon.estimator import LagLlamaEstimator
from gluonts.dataset.common import ListDataset
from torch.utils.data import Dataset
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddTimeFeatures,
    Chain,
    DummyValueImputation,
    Transformation,
)
from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.dataset.common import ListDataset




def create_listdataset_with_windows(X, seq_len=120, stride=120, freq='S', pred_length=30):
    """
    Crée un ListDataset avec des fenêtres à partir d'une série temporelle ordonnée X.

    Parameters:
    - X : numpy array représentant la série temporelle ordonnée chronologiquement.
    - seq_len : longueur de chaque fenêtre.
    - stride : pas entre les fenêtres.
    - freq : la fréquence des timestamps artificiels (par défaut 'S' pour secondes).
    - pred_length : longueur de la période de prédiction.

    Returns:
    - transformed_dataset : ListDataset transformé.
    """
    start_date = pd.Timestamp("2000-01-01") # Random start date
    if len(X.shape) == 1:
        windows = []
        for i in range(0, len(X) - seq_len + 1, stride):
            windows.append({
                "start": start_date + pd.to_timedelta(i, unit=freq),
                "target": X[i:i + seq_len]
            })
    elif len(X.shape) == 2:
        windows = []
        for i in range(X.shape[0]):
            windows.append({
                "start": start_date + pd.to_timedelta(i, unit=freq),
                "target": X[i]
            })
    dataset = ListDataset(windows, freq=freq)

    transform = Chain(
                [
                    AddTimeFeatures(
                        start_field=FieldName.START,
                        target_field=FieldName.TARGET,
                        output_field=FieldName.FEAT_TIME,
                        time_features=time_features_from_frequency_str("S"),
                        pred_length=pred_length,
                    ),
                    AddObservedValuesIndicator(
                        target_field=FieldName.TARGET,
                        output_field=FieldName.OBSERVED_VALUES,
                        imputation_method=DummyValueImputation(0.0),
                    ),
                ]
            )

    transformed_dataset = transform(dataset, is_train=True)

    return transformed_dataset


def generate_future_time_features(start_date, pred_length, freq='S'):
    """
    Génère les caractéristiques temporelles futures en utilisant la fonction time_features_from_frequency_str de GluonTS.

    Parameters:
    - start_date : pd.Timestamp, date de départ pour générer les caractéristiques.
    - pred_length : int, longueur de la période de prédiction.
    - freq : str, fréquence des timestamps (par défaut 'S' pour secondes).

    Returns:
    - future_time_feat : np.array, tableau des caractéristiques temporelles futures.
    """
    time_features = time_features_from_frequency_str(freq)

    future_dates = pd.date_range(start=start_date, periods=pred_length, freq=freq)

    num_features = len(time_features)
    future_time_feat = np.zeros((pred_length, num_features))

    for i, date in enumerate(future_dates):
        for j, feature in enumerate(time_features):
            future_time_feat[i, j] = feature(date)

    return future_time_feat


class LagLlamaTorchLoader(Dataset):
    def __init__(self, X, y, seq_len=120, stride=120, num_samples=100,
                 model_size=1124, pred_length=1, device="cuda"):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.stride = stride
        self.dataset = create_listdataset_with_windows(X, seq_len, stride, freq="S",
                                                       pred_length=pred_length)
        self.list_X = []
        for entry in self.dataset:
            self.list_X.append(entry)
        self.pred_length = pred_length
        self.num_samples = num_samples
        self.model_size = model_size
        self.device = device

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = torch.zeros(1,self.model_size)
        sample[0, -self.seq_len:] = torch.tensor(self.list_X[idx]["target"], dtype=torch.float)
        sample = sample.repeat_interleave(self.num_samples, 0)

        past_observed_value = torch.zeros(1,self.model_size)
        past_observed_value[0, -self.seq_len:] = torch.tensor(self.list_X[idx]["observed_values"],
                                                              dtype=torch.float)
        past_observed_value = past_observed_value.repeat_interleave(self.num_samples, 0)

        start_date = self.list_X[idx]["start"]

        future_time_feat = torch.tensor(generate_future_time_features(start_date.to_timestamp(),
                                        self.pred_length, "S"), dtype=torch.float)
        future_time_feat = future_time_feat.unsqueeze(0).repeat_interleave(self.num_samples, 0)

        time_feat = torch.zeros(1, self.model_size, 6)
        time_feat[0, -self.seq_len:, :] = torch.tensor(self.list_X[idx]["time_feat"],
                                                       dtype=torch.float).permute(1,0)
        time_feat = time_feat.repeat_interleave(self.num_samples, 0)

        return sample, past_observed_value, time_feat, future_time_feat


def load_llama(X, prediction_length=1, context_length=32, stride=32, num_samples=1,
               ckpt_path="lag-llama.ckpt"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]


    rope_scaling_arguments = {
        "type": "linear",
        "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
    }
    estimator = LagLlamaEstimator(
        ckpt_path=ckpt_path,
        prediction_length=prediction_length,
        context_length=context_length,
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        rope_scaling=None,

        batch_size=1,
        num_parallel_samples=num_samples,
        device=device,
    )
    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    dataset = LagLlamaTorchLoader(X, None, seq_len=context_length,
                                  stride=stride, num_samples=num_samples,
                                  pred_length=prediction_length)

    return predictor.network.model, dataset


def get_superposed_latent_representation(module, input, output):
    global latent_representations
    latent_representations.append(output)


def get_superposed_llama_representation(batch, model, device="cuda"):
    global latent_representations
    latent_representations = []

    hook_handles = []
    for block in model.transformer.h:
        hook_handles.append(block.rms_1.register_forward_hook(get_superposed_latent_representation))

    with torch.no_grad():
        _ = model(batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device))

    for handle in hook_handles:
        handle.remove()

    superposed_representation = torch.cat(latent_representations, dim=2)

    return superposed_representation


class LatentLlamaLoader(Dataset):
    def __init__(self, dataset, transform, model, device="cuda"):
        self.dataset = dataset
        self.transform = transform
        self.model = model
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self.transform(item, self.model, self.device)
