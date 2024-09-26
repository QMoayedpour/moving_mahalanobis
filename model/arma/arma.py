import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm
from utils.metrics import all_metrics
from utils.utils import score_windows
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import warnings
import os


class ArmAnomalyDetector:
    def __init__(self, X=None, y=None,
                 max_p=5, max_q=5, n_data=5000, verbose=False,
                 model_name="ARMA", seq_len=120,
                 windows=True, **kwargs):
        self.X = X
        self.y = y
        self.max_p = max_p
        self.max_q = max_q
        self.seq_len = seq_len
        self.windows = windows
        self.n_data = n_data
        self.verbose = verbose
        self.model = None
        self.residuals = None
        self.threshold = None
        self.anomalies = None
        self.score = None
        self.model_name = model_name

        for param_name, param_value in kwargs.items():
            setattr(self, param_name, param_value)

    def fit(self):
        best_aic = np.inf
        best_order = None
        best_model = None

        self.time_series = pd.Series(self.X).dropna().astype(float)

        def fit_arima(p, q):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    temp_model = sm.tsa.ARIMA(self.time_series.iloc[:self.n_data], order=(p, 0, q)).fit(method='innovations_mle')
                    temp_aic = temp_model.aic
                    return (p, q, temp_aic, temp_model)
            except:
                return (p, q, np.inf, None)

        p_range = range(self.max_p)
        q_range = range(self.max_q)
        parameter_grid = [(p, q) for p in p_range for q in q_range]

        if self.verbose:
            results = Parallel(n_jobs=-1)(
                delayed(fit_arima)(p, q) for p, q in tqdm(parameter_grid, desc="Fitting models")
            )
        else:
            results = Parallel(n_jobs=-1)(
                delayed(fit_arima)(p, q) for p, q in parameter_grid
            )

        for result in results:
            p, q, temp_aic, temp_model = result

            if temp_model is not None and temp_aic < best_aic:
                best_aic = temp_aic
                best_order = (p, q)

        self.model = sm.tsa.ARIMA(self.time_series[:], order=(best_order[0], 0, best_order[1])).fit()

        self.residuals = self.model.resid
        self.score = np.abs(self.model.resid)
        self.arma_order = best_order

        self.labels = np.array(self.y).flatten()[-len(self.score):]

        if self.windows:
            labels, score = score_windows(self.labels, self.score, seq_len=self.seq_len)
        else:
            labels, score = self.labels, self.score

        out = all_metrics(labels, score, model=self.model_name)

        return out

    def detect_anomalies(self, true_labels):
        """detect_anomalies and choosing the threshold that maximise the f1 score

        Args:
            true_labels (array): array of true predictions

        Raises:
            ValueError: No model fitted yet, use .fit()

        Returns:
            array: Anomalies labelled data for optimal f1 scores
        """
        if self.residuals is None:
            raise ValueError("Model has not been fitted yet, use .fit().")

        best_f1 = 0
        best_threshold = 0
        best_anomalies = None

        for threshold in np.linspace(min(np.abs(self.residuals)), max(np.abs(self.residuals)), 100):
            anomalies = np.abs(self.residuals) > threshold
            f1 = f1_score(true_labels, anomalies)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_anomalies = anomalies

        self.threshold = best_threshold
        self.anomalies = best_anomalies
        return self.anomalies

    def detect_anomalies_ratio(self, anomaly_ratio):
        """Detect anomalie for a given ratio of anomalies

        Args:
            anomaly_ratio (float): anomalie ratio (from 0 to 1)

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            array: Anomalies labelled data for optimal anomaly_ratio part of anomalies
        """
        if self.residuals is None:
            raise ValueError("Model has not been fitted yet.")
        if not (0 <= anomaly_ratio <= 1):
            raise ValueError("Anomaly ratio must be between 0 and 1.")

        self.threshold = np.quantile(np.abs(self.residuals), 1 - anomaly_ratio)
        self.anomalies = np.abs(self.residuals) > self.threshold
        return self.anomalies

    def _score(self, true_labels):
        if self.anomalies is None:
            raise ValueError("Anomalies have not been detected yet.")

        accuracy = accuracy_score(true_labels, self.anomalies)
        precision = precision_score(true_labels, self.anomalies)
        recall = recall_score(true_labels, self.anomalies)
        f1 = f1_score(true_labels, self.anomalies)
        roc_auc = roc_auc_score(true_labels, self.score)

        precision_values, recall_values, _ = precision_recall_curve(true_labels, self.score)
        aupr = auc(recall_values, precision_values)

        sorted_indices = np.argsort(-self.score)
        sorted_true_labels = true_labels[sorted_indices]
        cumsum_true_labels = np.cumsum(sorted_true_labels)
        recall_95_threshold = np.where(cumsum_true_labels >= 0.95 * np.sum(true_labels))[0][0]
        fpr_95 = np.sum((1 - true_labels)[:recall_95_threshold]) / np.sum(1 - true_labels)
        p, q = self.arma_order

        return {
            'accuracy': accuracy,
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'aupr': float(aupr),
            'fpr_95': float(fpr_95),
            'n_anomaly': int(true_labels.sum()),
            'n_data': len(self.time_series),
            'Model': f"ARMA {p},{q}"
        }

    def plot_anomalies(self, true_labels):
        if self.anomalies is None:
            raise ValueError("Anomalies have not been detected yet.")

        plt.figure(figsize=(14, 8))
        plt.plot(self.time_series.index, self.time_series.values, label='serie', c='blue')
        true_anomalies = np.where(true_labels == 1)[0]
        plt.scatter(self.time_series.index[true_anomalies], self.time_series.values[true_anomalies],
                    color='red', label='True Anomalies', s=100, marker='X')

        detected_anomalies = np.where(self.anomalies)[0]
        plt.scatter(self.time_series.index[detected_anomalies],
                    self.time_series.values[detected_anomalies],
                    color='orange', label='Detected Anomalies', s=50, marker='o')

        plt.title('')
        plt.xlabel('time')
        plt.ylabel('value')
        plt.legend()
        plt.grid(True)
        plt.show()