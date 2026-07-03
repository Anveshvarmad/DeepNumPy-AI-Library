import numpy as np

from ..core import Tensor
from ..core import Parameter


class Module:
    def __init__(self):
        self.training = True

    def forward(self, *inputs):
        raise NotImplementedError

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def parameters(self):
        params = []

        def collect(value):
            if isinstance(value, Parameter):
                params.append(value)

            elif isinstance(value, Module):
                params.extend(value.parameters())

            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)

            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)

        for value in self.__dict__.values():
            collect(value)

        return params

    def children(self):
        modules = []

        def collect(value):
            if isinstance(value, Module):
                modules.append(value)

            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)

            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)

        for value in self.__dict__.values():
            collect(value)

        return modules

    def train(self):
        self.training = True
        for module in self.children():
            module.train()
        return self

    def eval(self):
        self.training = False
        for module in self.children():
            module.eval()
        return self

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def state_dict(self):
        state = {}

        def collect(prefix, module):
            for name, value in module.__dict__.items():
                if isinstance(value, Parameter):
                    key = f"{prefix}.{name}" if prefix else name
                    state[key] = value.data.copy()

                elif isinstance(value, Module):
                    key = f"{prefix}.{name}" if prefix else name
                    collect(key, value)

                elif isinstance(value, (list, tuple)):
                    for index, item in enumerate(value):
                        if isinstance(item, Module):
                            key = f"{prefix}.{name}.{index}" if prefix else f"{name}.{index}"
                            collect(key, item)

        collect("", self)
        return state

    def load_state_dict(self, state):
        params = {}

        def collect(prefix, module):
            for name, value in module.__dict__.items():
                if isinstance(value, Parameter):
                    key = f"{prefix}.{name}" if prefix else name
                    params[key] = value

                elif isinstance(value, Module):
                    key = f"{prefix}.{name}" if prefix else name
                    collect(key, value)

                elif isinstance(value, (list, tuple)):
                    for index, item in enumerate(value):
                        if isinstance(item, Module):
                            key = f"{prefix}.{name}.{index}" if prefix else f"{name}.{index}"
                            collect(key, item)

        collect("", self)

        for key, value in state.items():
            if key not in params:
                continue

            if params[key].data.shape != value.shape:
                raise ValueError(
                    f"Shape mismatch for {key}: expected {params[key].data.shape}, got {value.shape}"
                )

            params[key].data[...] = value

    def save(self, path):
        np.savez(path, **self.state_dict())

    def load(self, path):
        loaded = np.load(path)
        state = {key: loaded[key] for key in loaded.files}
        self.load_state_dict(state)

    def __repr__(self):
        child_lines = []

        for name, value in self.__dict__.items():
            if isinstance(value, Module):
                child_lines.append(f"  ({name}): {value}")

            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    if isinstance(item, Module):
                        child_lines.append(f"  ({name}.{index}): {item}")

        if not child_lines:
            return f"{self.__class__.__name__}()"

        return f"{self.__class__.__name__}(\n" + "\n".join(child_lines) + "\n)"


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            (in_features, out_features),
            init="xavier",
            name="weight"
        )

        self.bias = Parameter(
            (1, out_features),
            init="zeros",
            name="bias"
        ) if bias else None

    def forward(self, x):
        output = x @ self.weight

        if self.bias is not None:
            output = output + self.bias

        return output

    def __repr__(self):
        return (
            f"Linear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None})"
        )


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.relu()


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.sigmoid()


class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.tanh()


class Softmax(Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return x.softmax(axis=self.axis)


class Flatten(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()

        if p < 0 or p >= 1:
            raise ValueError("dropout probability must be in [0, 1)")

        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        mask = (np.random.rand(*x.shape) > self.p).astype(np.float64)
        mask = mask / (1.0 - self.p)

        return x * Tensor(mask, requires_grad=False)

    def __repr__(self):
        return f"Dropout(p={self.p})"


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def __getitem__(self, index):
        return self.layers[index]

    def __len__(self):
        return len(self.layers)

    def append(self, layer):
        self.layers.append(layer)

    def __repr__(self):
        lines = ["Sequential("]

        for index, layer in enumerate(self.layers):
            lines.append(f"  ({index}): {layer}")

        lines.append(")")
        return "\n".join(lines)


class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        self.modules_list = list(modules) if modules is not None else []

    def append(self, module):
        self.modules_list.append(module)

    def __getitem__(self, index):
        return self.modules_list[index]

    def __len__(self):
        return len(self.modules_list)

    def __iter__(self):
        return iter(self.modules_list)


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = normalized_shape
        self.eps = eps

        self.gamma = Parameter(normalized_shape, init="ones", name="gamma")
        self.beta = Parameter(normalized_shape, init="zeros", name="beta")

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        normalized = (x - mean) / ((variance + self.eps) ** 0.5)

        return self.gamma * normalized + self.beta

    def __repr__(self):
        return f"LayerNorm(normalized_shape={self.normalized_shape})"


class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = Parameter((1, num_features), init="ones", name="gamma")
        self.beta = Parameter((1, num_features), init="zeros", name="beta")

        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(axis=0, keepdims=True)
            variance = ((x - mean) ** 2).mean(axis=0, keepdims=True)

            self.running_mean = (
                (1 - self.momentum) * self.running_mean
                + self.momentum * mean.data
            )

            self.running_var = (
                (1 - self.momentum) * self.running_var
                + self.momentum * variance.data
            )

        else:
            mean = Tensor(self.running_mean, requires_grad=False)
            variance = Tensor(self.running_var, requires_grad=False)

        normalized = (x - mean) / ((variance + self.eps) ** 0.5)

        return self.gamma * normalized + self.beta

    def __repr__(self):
        return f"BatchNorm1d(num_features={self.num_features})"
