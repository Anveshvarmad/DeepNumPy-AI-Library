import numpy as np

from ..core import Tensor


class Dataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        if len(tensors) == 0:
            raise ValueError("TensorDataset requires at least one tensor")

        self.tensors = []

        for tensor in tensors:
            if isinstance(tensor, Tensor):
                self.tensors.append(tensor.data)
            else:
                self.tensors.append(np.array(tensor, dtype=np.float64))

        first_length = len(self.tensors[0])

        for tensor in self.tensors:
            if len(tensor) != first_length:
                raise ValueError("All tensors must have the same first dimension")

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, index):
        items = []

        for tensor in self.tensors:
            items.append(Tensor(tensor[index], requires_grad=False))

        if len(items) == 1:
            return items[0]

        return tuple(items)


def random_split(dataset, lengths, seed=None):
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of split lengths must equal dataset length")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))

    splits = []
    start = 0

    for length in lengths:
        split_indices = indices[start:start + length]
        splits.append(Subset(dataset, split_indices))
        start += length

    return tuple(splits)


class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]
