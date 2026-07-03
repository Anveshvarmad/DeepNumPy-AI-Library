import numpy as np

from ..core import Tensor


def _to_numpy(value):
    if isinstance(value, Tensor):
        return value.data

    return np.array(value)


def accuracy(prediction, target):
    prediction = _to_numpy(prediction)
    target = _to_numpy(target)

    if target.ndim == 2 and target.shape[1] == 1:
        target = target.reshape(-1)

    if prediction.ndim == 2 and prediction.shape[1] > 1:
        predicted_labels = prediction.argmax(axis=1)
    else:
        predicted_labels = (prediction.reshape(-1) >= 0.5).astype(np.int64)

    target = target.reshape(-1).astype(np.int64)

    return float((predicted_labels == target).mean())


def binary_accuracy(prediction, target, threshold=0.5):
    prediction = _to_numpy(prediction).reshape(-1)
    target = _to_numpy(target).reshape(-1)

    predicted_labels = (prediction >= threshold).astype(np.int64)
    target = target.astype(np.int64)

    return float((predicted_labels == target).mean())


def mean_absolute_error(prediction, target):
    prediction = _to_numpy(prediction)
    target = _to_numpy(target)

    return float(np.abs(prediction - target).mean())


def mean_squared_error(prediction, target):
    prediction = _to_numpy(prediction)
    target = _to_numpy(target)

    return float(((prediction - target) ** 2).mean())
