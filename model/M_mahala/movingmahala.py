import numpy as np
import torch
from utils.utils import (get_coreset_idx,
                         create_windows,
                         group_cons_num,
                         plot_label,
                         get_modif_score)
from utils.metrics import all_metrics, plot_roc
from scipy.spatial.distance import mahalanobis
from tqdm import tqdm


class MovingMahalanobis:
    def __init__(self, y, dataloader, model_name="LagLlama", X=None, seq_len=32):
        self.y = y
        self.model_name = model_name
        self.dataloader = dataloader
        self.seq_len = seq_len
        self.X = X.flatten()[(10 + 1) * seq_len:] if isinstance(X, np.ndarray) else X
        self.trained = True
        self.score = None
        self.labels = None

    def fit(self, print_results=False, thresh=10, verbose=False, modif_score=False,
            windows=False, selection="random", n_channels=16, mode_auto=True):

        assert selection in ["random", "coreset", "n_firsts"]
        if self.model_name == "TS2Vec":
            selection = "n_firsts"

        list_vec, dist_list, list_y = [], [], []
        bar = tqdm(zip(self.dataloader, self.y), total=self.y.shape[0]) if verbose else zip(self.dataloader, self.y)

        for i, (x, y) in enumerate(bar, start=1):
            if i == 1:
                selected_idx = list(range(x.shape[2]))
            representation = x[:, :, selected_idx]
            list_vec.append(torch.tensor(representation.reshape(representation.shape[0],
                                                                representation.shape[2], -1),
                                                                dtype=torch.float))
            if len(list_vec) < thresh:
                continue

            if len(list_vec) == thresh:
                selected_idx = self._select_indices(torch.cat(list_vec, dim=0), selection, n_channels)
                list_vec = [tensor[:, selected_idx, :] for tensor in list_vec]

            list_y.append(y)
            means, covs = self._calculate_means_covariance(torch.cat(list_vec, dim=0)[:-1,:,:])
            dist_list.extend(self._calculate_distances(means, covs, list_vec[-1].cpu().detach().numpy()))

        self.score = np.array(dist_list).squeeze(1)

        self.labels = np.concatenate(list_y)
        self.score = self._apply_modified_score(modif_score, n_channels)
        results = self._evaluate_results(self.score)

        if print_results:
            self._print_results(results, verbose)

        if windows:
            return self._calculate_windows_results()

        return results

    def _select_indices(self, vec, selection, n_channels):
        if selection == "coreset":
            return get_coreset_idx(vec[0].permute(1, 0).cpu(), eps=0.9, n=n_channels)
        elif selection == "random":
            return np.random.choice(vec.shape[1], n_channels, replace=False)
        elif selection == "n_firsts":
            return np.arange(n_channels)

    def _calculate_means_covariance(self, vec):
        B, C, L = vec.shape
        means = torch.mean(vec, dim=0).cpu().numpy()
        covs = np.zeros((C, C, L))

        for i in range(L):
            covs[:, :, i] = np.cov(vec[:, :, i].cpu().numpy(), rowvar=False, ddof=1) + 0.001 * np.eye(C)

        return means, covs

    def _calculate_distances(self, means, covs, samples):
        cov_invs = np.zeros_like(covs)
        epsilon = 1e-3
        k = covs[:, :, 0].shape[0]
        diagonal_values = np.full(k, epsilon)
        diagonal_matrix = np.diag(diagonal_values)

        for i in range(cov_invs.shape[2]):
            cov_invs[:, :, i] = np.linalg.inv(covs[:, :, i] + diagonal_matrix)
        return [np.apply_along_axis(lambda x: mahalanobis(x, means[:, i], cov_invs[:, :, i]), 1, samples[:, :, i]) for i in range(samples.shape[2])]

    def _apply_modified_score(self, modif_score, n_channels=16):
        if modif_score:
            return get_modif_score(self.labels[:], self.score)
        return self.score ** 2 * self._get_adjustor(n_channels) if self.score is not None else self.score

    def _get_adjustor(self, n_channels):
        adjustor = [max((i - n_channels - 2) / (i - 1), 0 )for i in range(n_channels, len(self.score) // self.seq_len)]
        adjustor = [x for x in adjustor for _ in range(self.seq_len)]
        return np.array(adjustor + [1] * (self.score.shape[0] - len(adjustor)))

    def _evaluate_results(self, score):
        results = all_metrics(self.labels[:], score[:])
        results.update({
            "n_data": self.labels.shape[0],
            "n_anomaly": int(self.labels.sum()),
            "model": self.model_name
        })
        return results

    def _print_results(self, results, verbose):
        if verbose:
            plot_roc(self.labels[:], self.score)
        print(results)

    def _calculate_windows_results(self):
        scores = create_windows(self.score[:], seq_len=self.seq_len, stride=self.seq_len)
        labels = create_windows(self.labels[:], seq_len=self.seq_len, stride=self.seq_len)
        max_scores = np.max(scores, axis=1)
        max_labels = np.max(labels, axis=1)

        results_windows = all_metrics(max_labels[:], max_scores[:])
        results_windows.update({
            "n_data": max_scores.shape[0],
            "n_anomaly": int(max_labels.sum()),
            "model": self.model_name
        })
        return results_windows

    def plot_anomalies(self, index=1, border=100):
        indices = np.where(self.labels.flatten() == 1)[0]
        array_indices = group_cons_num(indices)
        start = array_indices[index][0] - border
        end = array_indices[index][-1] + border
        plot_label(self.labels[start:end], self.score[start:end], serie=self.X[start:end] * 16, seq_len=self.seq_len)
