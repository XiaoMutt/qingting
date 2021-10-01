import numpy as np
import torch

from core.config import DEVICE


def np2torch(x):
    x = torch.from_numpy(x).to(DEVICE)
    return x


def average_precision(prediction: np.ndarray, oracle: np.ndarray, threshold: float) -> float:
    """
    Return the average precision
    :param prediction: prediction probability array
    :param oracle: oracle array
    :param threshold: a float above which the prediction is considered positive
    :return: average precision
    """
    positive_mask = prediction > threshold
    prediction_ = prediction[positive_mask]
    oracle_ = oracle[positive_mask]

    num_oracle_true = oracle_.sum()
    if num_oracle_true == 0:
        return 0

    sorted_indices = np.argsort(-prediction_)
    oracle_ = np.take_along_axis(oracle_, sorted_indices, axis=0)

    recall_array = np.zeros(num_oracle_true + 1)
    num_real_positive_array = np.cumsum(oracle_)

    for num_predicted_positive, num_real_positive in enumerate(num_real_positive_array, start=1):
        precision = num_real_positive / num_predicted_positive
        recall_array[num_real_positive] = max(recall_array[num_real_positive], precision)

    res = recall_array.sum() / num_oracle_true
    return res
