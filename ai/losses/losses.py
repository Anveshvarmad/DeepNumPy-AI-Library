import numpy as np

from ..core import Tensor


def _ensure_tensor(value):
    if isinstance(value, Tensor):
        return value

    return Tensor(value, requires_grad=False)


class Loss:
    def __call__(self, prediction, target):
        return self.forward(prediction, target)

    def forward(self, prediction, target):
        raise NotImplementedError


class MSELoss(Loss):
    def forward(self, prediction, target):
        target = _ensure_tensor(target)
        return ((prediction - target) ** 2).mean()


class L1Loss(Loss):
    def forward(self, prediction, target):
        target = _ensure_tensor(target)
        return (prediction - target).abs().mean()


class BCELoss(Loss):
    def __init__(self, eps=1e-8):
        self.eps = eps

    def forward(self, prediction, target):
        target = _ensure_tensor(target)

        return -(
            target * (prediction + self.eps).log()
            + (1 - target) * (1 - prediction + self.eps).log()
        ).mean()


class CrossEntropyLoss(Loss):
    def forward(self, logits, target):
        if isinstance(target, Tensor):
            target = target.data

        target = np.array(target, dtype=np.int64)

        if target.ndim == 2 and target.shape[1] == 1:
            target = target.reshape(-1)

        if logits.data.ndim != 2:
            raise ValueError("CrossEntropyLoss expects logits with shape (batch_size, num_classes)")

        batch_size = logits.data.shape[0]
        log_probs = logits.log_softmax(axis=1)

        selected = log_probs[np.arange(batch_size), target]

        return -selected.mean()


class NLLLoss(Loss):
    def forward(self, log_probs, target):
        if isinstance(target, Tensor):
            target = target.data

        target = np.array(target, dtype=np.int64)

        if target.ndim == 2 and target.shape[1] == 1:
            target = target.reshape(-1)

        batch_size = log_probs.data.shape[0]
        selected = log_probs[np.arange(batch_size), target]

        return -selected.mean()


class HuberLoss(Loss):
    def __init__(self, delta=1.0):
        self.delta = delta

    def forward(self, prediction, target):
        target = _ensure_tensor(target)

        error = prediction - target
        abs_error = error.abs()

        mask = Tensor(
            (abs_error.data <= self.delta).astype(np.float64),
            requires_grad=False
        )

        quadratic = 0.5 * (error ** 2)
        linear = self.delta * (abs_error - 0.5 * self.delta)

        return (mask * quadratic + (1 - mask) * linear).mean()
