import math
import numpy as np

from ..core import Tensor


class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=False, drop_last=False, seed=None):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size

        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        indices = np.arange(len(self.dataset))

        if self.shuffle:
            self.rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]

            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            samples = [self.dataset[int(index)] for index in batch_indices]

            yield self._collate(samples)

    def _collate(self, samples):
        first = samples[0]

        if isinstance(first, tuple):
            columns = list(zip(*samples))
            return tuple(self._stack(column) for column in columns)

        return self._stack(samples)

    def _stack(self, values):
        arrays = []

        for value in values:
            if isinstance(value, Tensor):
                arrays.append(value.data)
            else:
                arrays.append(np.array(value, dtype=np.float64))

        return Tensor(np.stack(arrays, axis=0), requires_grad=False)
