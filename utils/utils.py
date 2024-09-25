import torch
from torch import tensor
from sklearn import random_projection
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import json


def get_coreset_idx(
    z_lib : tensor, 
    n : int = 1000,
    eps : float = 0.90,
    float16 : bool = True,
    force_cpu : bool = False,
    verbose : bool = False,
) -> tensor:
    """
        In order to obtain a reduced version of the Memory bank we use this function that will perform a greedy coreset supsampling. 
        The memory bnak become then fully searchable for larger image size and counts, allowing for patch-based comparison beneficial to anomaly detection. 
        With random subsampling, some significant information will be losed in the coverage of nominal features. 
        The coreset subsampling mechanism implemented in this method reduces the Memory bank in a better way and reduces inference time. 

        Conceptually, the selection of basic groups aims to find a subset so that the solutions of problems on A can be more closely 
        and above all more quickly approached by those calculated on S. 

        5 parameters characterized this function : 

        - tensor_list : This is a tensor representing a list of tensors that will be subsampled

        - n : Corresponds to the number of tensors that will be kept in the subsampling. It has been calculated based on a percentage given (the percentage of the Memory bank
        that we want to keep) multiplied by the number of tensors initially in the Memory bank

        - eps : This is a float that corresponds to the epsilon value that will be used in the random projection 

        - float16 : Boolean to determine if we want to use the 16-bit float precision or not

        - force_cpu: Boolean to determine if we want to use the CPU for the computations or not

        To better understand the following algorithm, first a random projection is performed on the tensor_list, then a list that will store indices is initialized
        The algorithm iterates from 0 to n -1 :
        - If this is the first iteration, the last_item and min_distances variables are initialized
        - For the rest of the iterations, for each one, the precedent variables are updates based on the distance between the rows of the tensor_list and last_item. 
          Then the row with the maximum value in min_distances is selected and it index is added to the list of indices. 
    """
    if verbose:
        print("Beginning of the coreset subsampling reduction...")

        print(f"   Fitting random projections. Start dim = {z_lib.shape}.")
    try:
        # A random projection is performed on the tensor_list
        transformer = random_projection.SparseRandomProjection(eps=eps)
        z_lib = torch.tensor(transformer.fit_transform(z_lib))
        if verbose:
            print(f"   DONE.                 Transformed dim = {z_lib.shape}.")
        if z_lib.shape[1]<n:
            print("NIMPORTE QOI!!!!")

    except ValueError:
        print("   Error: could not project vectors. Please increase `eps`.")
        return torch.randperm(n + 1)

    select_idx = 0
    last_item = z_lib[select_idx:select_idx+1]
    coreset_idx = [torch.tensor(select_idx)]
    min_distances = torch.linalg.norm(z_lib-last_item, dim=1, keepdims=True)

    if float16:
        last_item = last_item.half()
        z_lib = z_lib.half()
        min_distances = min_distances.half()
    if torch.cuda.is_available() and not force_cpu:
        last_item = last_item.to("cuda")
        z_lib = z_lib.to("cuda")
        min_distances = min_distances.to("cuda")

    # Iteration from 0 to n-1, the number of tensors that will be kept in the subsampling
    for i in range(n-1):

        # The variables are updates based on the distances calculations
        # Broadcasting step
        distances = torch.linalg.norm(z_lib-last_item, dim=1, keepdims=True) 
        # Iterative step
        min_distances = torch.minimum(distances, min_distances) 
        # Selection step
        select_idx = torch.argmax(min_distances) 

        # bookkeeping
        last_item = z_lib[select_idx:select_idx+1]
        min_distances[select_idx] = 0
        coreset_idx.append(select_idx.to("cpu"))

    if verbose:
        print("End of the coreset subsampling reduction")
    return torch.stack(coreset_idx)


def split_anomaly(data_anomaly, seq_len=120, test_size=0.5, random_state=42, stride=1):
    X_large = data_anomaly.value.to_numpy()
    y_large = data_anomaly.label.to_numpy()

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

    # Split data

    scaler = MinMaxScaler()
    X_shape = X.shape
    X = scaler.fit_transform(X.reshape(-1, X_shape[-1])).reshape(X_shape)

    X_train, X_test, y_train, y_test, y_train_mask, y_test_mask = train_test_split(
        X, y, y_mask, test_size=test_size, random_state=random_state
    )
    class_to_keep, class_to_remove = 0,  1

    mask_train = y_train_mask == class_to_keep
    mask_train_remove = y_train_mask >= class_to_remove

    X_train_healthy = X_train[mask_train]
    y_train_healthy = y_train[mask_train]

    X_train_anomaly = X_train[mask_train_remove]
    y_train_anomaly = y_train[mask_train_remove]

    return (X_train_healthy, y_train_healthy, np.concatenate((X_train_anomaly, X_test), axis=0),
            np.concatenate((y_train_anomaly, y_test), axis=0))


def get_modif_score(y_true, score):
    """
    Modified metrics;that is, identify the anomaly by segment, not a point.
    If any point in an anomaly segment in the ground truth can be detected by a chosen threshold,
    we say this segment is detected correctly, and all points in this segment are treated as if they can be detected
    by this threshold.
    Examples
    ----------
    y_true =    [1,      1,      1,       0, 0, 0,   1,    1,      1,      1,     0,    1]
    score  =    [-0.1,  -0.3,   -0.1,     0, 0, 0,   0,    0,      0,     -0.5,   0,    0]

    modified =  [-0.3,  -0.3,   -0.3,     0, 0, 0,  -0.5,  -0.5,  -0.5,   -0.5,   0,    0]
    Parameters
    ----------
    score: 1-D np.array
    y_true: 1-D np.array

    Returns
    -------
    1-D np.array

    """

    y_true = np.asarray(y_true, dtype=float) # [1, 0, 1, 1, 1, 0, 0]
    score = np.asarray(score, dtype=float) # [0.5, 0.2, 0.5, 0.8, 0.1, 0.6, 0.2]
                                            # [0.5, 0.2, 0.8, 0.8, 0.8, 0.6, 0.2]

    assert y_true.shape[0] == score.shape[0]
    assert len(y_true.shape) == 1
    assert len(score.shape) == 1
    modified_score = score.copy()

    time_spans = []
    size_of_y = len(y_true)
    _s = 0
    _n = 0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            _n = i + 1
            if i == size_of_y - 1:
                time_spans.append(np.arange(start=_s, stop=_n, step=1))
        else:
            if _n > _s:
                time_spans.append(np.arange(start=_s, stop=_n, step=1))
            _s = i + 1
    for ts in time_spans:
        modified_score[ts] = np.repeat(np.max(modified_score[ts]), len(ts))
    return modified_score


def comput_modif_score(labels, score):
    m_score = get_modif_score(labels[:, -1], score[:, -1])
    return labels[:, -1], m_score


def save_to_json(dictionary, file_path, verbose=False):
    with open(file_path, 'w') as json_file:
        json.dump(dictionary, json_file, indent=4, ensure_ascii=False)
    if verbose:
        print(f"results saved @ {file_path}")


def create_windows(array, seq_len=120, stride=120):
    array_list = []
    for i in range(0, len(array) - seq_len, stride):
        array_list.append(array[i:i+seq_len])
    return np.array(array_list)


def score_windows(labels, score, seq_len=120):
    labels = create_windows(labels, seq_len=seq_len, stride=seq_len)
    score = create_windows(score, seq_len=seq_len, stride=seq_len)

    return np.max(labels, axis=1), np.max(score, axis=1)


def plot_label(arr, score, score_2=None, serie=None, seq_len=120, grid=True, save_fig=False):
    anomaly_indices = np.where(arr == 1)[0]
    fig, ax = plt.subplots(figsize=(20, 5))

    if grid:
        for i in range(0, len(arr), seq_len):
            ax.axvline(x=i, color='grey', linestyle='--', linewidth=1)

    if len(anomaly_indices) > 0:
        ax.axvline(x=anomaly_indices[0], color='red', linestyle='-', alpha=0.5,
                   linewidth=1, label="Anomaly")
        for index in anomaly_indices[1:]:
            ax.axvline(x=index, color='red', linestyle='-', alpha=0.5, linewidth=1)

    ax.plot(score, label="Anomaly Score", linestyle='-', c="blue")
    ax.set_ylabel('', color="b")
    ax.tick_params(axis='y', labelcolor="b")

    if score_2 is not None:
        ax.plot(score_2, label="Modified Score", linestyle='-')

    if serie is not None:
        ax2 = ax.twinx()
        ax2.plot(serie, label="Original Serie", color='black', linestyle='-', lw=0.8)
        ax2.yaxis.tick_right()

    fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85), ncol=1)
    if save_fig:
        plt.savefig(save_fig)
    plt.show()


def group_cons_num(arr):
    if arr.size == 0:
        return np.array([], dtype=object)

    diff = np.diff(arr) != 1

    split_indices = np.where(diff)[0] + 1

    split_arrays = np.split(arr, split_indices)
    
    return np.array(split_arrays, dtype=object)


def extract_dataset_name(text):
    start_index = text.rfind('/')
    end_index = text.rfind('.csv')
    if start_index != -1 and end_index != -1 and start_index < end_index:
        return text[start_index + 1:end_index]
    return None


def adjust_score(arr, k=8, l=120):
    """Adju Score Function:
    Take an array or list (arr) and adjust the score by subscrapting the mean of the k last point
    spaced from l.

    arr (np.array): array 1d or list of the scores
    k (int): how many values to take to calculate the mean
    l (int): space between the points to calculate the mean

    for x_i in arr:
        x_i' <- x_i - k**-1 *sum(j=1 to k) x_i-j*l
    """
    arr = np.array(arr)
    n = len(arr)

    k += 1

    result = np.zeros(n)

    for i in range(n):

        indices = np.arange(i - l, i - l * k, -l)
        indices = indices[indices >= 0]

        if len(indices) > 0:

            mean_previous_points = np.mean(arr[indices])
            result[i] = (arr[i] - mean_previous_points)#/ mean_previous_points if mean_previous_points != 0 else (arr[i] - mean_previous_points)/ (mean_previous_points+1)
        else:

            result[i] = arr[i]

    return result


def split_arrays_ano(X_large, y_large, seq_len=120, stride=120, test_size=0.5, random_state=42,
                     split=False):
    """take arrays to split subarrays for test and train. Train contains 0 sub array with anomaly

    Args:
        X_large (np.array): Value 1d array
        y_large (np.array): Labels 1d array
        seq_len (int): Size of sub arrays
        stride (int): stride
        split (bool): to put anomalies or not in the train set

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


def flatten_data(data):
    new_dic = {}
    for key in data.keys():
        X_train = np.array(data[key]["X_train"])
        X_test = np.array(data[key]["X_test"])

        y_train = np.array(data[key]["y_train"])
        y_test = np.array(data[key]["y_test"])

        new_dic[key] = {"X": np.concatenate([X_train, X_test]).flatten(),
                        "y": np.concatenate([y_train, y_test]).flatten()}
    return new_dic


def normalise(array, to_list=False):
    arr = np.array(array)
    normalised = (arr - arr.min())/(arr.max() - arr.min())
    return normalised.tolist() if to_list else normalised


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def push_json(output_path, dic):
    path = output_path
    with open(path, 'w') as json_file:
        json.dump(dic, json_file, indent=4)
    print(f"results saved @ {path}")


def load_data(dataset="NAB"):

    with open(f"./data/{dataset}_anomaly/{dataset}.json", "r") as file:
        data = json.load(file)

    return data
