import numpy as np


_grad_enabled = True


def manual_seed(seed):
    np.random.seed(seed)


def is_grad_enabled():
    return _grad_enabled


class no_grad:
    def __enter__(self):
        global _grad_enabled
        self.prev = _grad_enabled
        _grad_enabled = False

    def __exit__(self, exc_type, exc_value, traceback):
        global _grad_enabled
        _grad_enabled = self.prev


def _to_numpy(data):
    if isinstance(data, Tensor):
        return data.data
    return np.array(data, dtype=np.float64)


def _ensure_tensor(value):
    if isinstance(value, Tensor):
        return value
    return Tensor(value, requires_grad=False)


def _unbroadcast(grad, shape):
    if shape == ():
        return np.array(grad).sum()

    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)

    for axis, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad


def _normalize_axis(axis, ndim):
    if axis is None:
        return None

    if isinstance(axis, int):
        axis = (axis,)

    return tuple(a if a >= 0 else ndim + a for a in axis)


def stack(tensors, axis=0):
    tensors = [_ensure_tensor(tensor) for tensor in tensors]

    data = np.stack([tensor.data for tensor in tensors], axis=axis)
    requires_grad = any(tensor.requires_grad for tensor in tensors)

    out = Tensor(
        data,
        requires_grad=requires_grad,
        _children=tuple(tensors),
        _op="stack"
    )

    def _backward():
        for index, tensor in enumerate(tensors):
            if tensor.requires_grad:
                tensor.grad += np.take(out.grad, index, axis=axis)

    out._backward = _backward
    return out


class Tensor:
    __array_priority__ = 100

    def __init__(self, data, requires_grad=False, name=None, _children=(), _op=""):
        self.data = _to_numpy(data)
        self.requires_grad = bool(requires_grad and _grad_enabled)
        self.grad = np.zeros_like(self.data) if self.requires_grad else None

        self.name = name
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    def item(self):
        return self.data.item()

    def detach(self):
        return Tensor(self.data.copy(), requires_grad=False)

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def backward(self, grad=None):
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be provided for non-scalar tensors")
            grad = np.ones_like(self.data)
        else:
            grad = np.array(grad, dtype=np.float64)

        topo = []
        visited = set()

        def build(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build(child)
                topo.append(node)

        build(self)

        for node in topo:
            if node.requires_grad:
                node.grad = np.zeros_like(node.data)

        self.grad = grad

        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        other = _ensure_tensor(other)
        out = Tensor(
            self.data + other.data,
            self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="add"
        )

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        out = Tensor(
            -self.data,
            self.requires_grad,
            _children=(self,),
            _op="neg"
        )

        def _backward():
            if self.requires_grad:
                self.grad += -out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-_ensure_tensor(other))

    def __rsub__(self, other):
        return _ensure_tensor(other) + (-self)

    def __mul__(self, other):
        other = _ensure_tensor(other)
        out = Tensor(
            self.data * other.data,
            self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="mul"
        )

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = _ensure_tensor(other)
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return _ensure_tensor(other) * (self ** -1)

    def __pow__(self, power):
        out = Tensor(
            self.data ** power,
            self.requires_grad,
            _children=(self,),
            _op="pow"
        )

        def _backward():
            if self.requires_grad:
                self.grad += power * (self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = _ensure_tensor(other)
        out = Tensor(
            self.data @ other.data,
            self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="matmul"
        )

        def _backward():
            if self.requires_grad:
                grad_self = out.grad @ np.swapaxes(other.data, -1, -2)
                self.grad += _unbroadcast(grad_self, self.data.shape)

            if other.requires_grad:
                grad_other = np.swapaxes(self.data, -1, -2) @ out.grad
                other.grad += _unbroadcast(grad_other, other.data.shape)

        out._backward = _backward
        return out

    @property
    def T(self):
        out = Tensor(
            np.swapaxes(self.data, -1, -2),
            self.requires_grad,
            _children=(self,),
            _op="transpose"
        )

        def _backward():
            if self.requires_grad:
                self.grad += np.swapaxes(out.grad, -1, -2)

        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(
            self.data.reshape(*shape),
            self.requires_grad,
            _children=(self,),
            _op="reshape"
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def flatten(self):
        return self.reshape(self.data.shape[0], -1)

    def sum(self, axis=None, keepdims=False):
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            self.requires_grad,
            _children=(self,),
            _op="sum"
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad
                axes = _normalize_axis(axis, self.data.ndim)

                if axes is not None and not keepdims:
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, ax)

                self.grad += np.ones_like(self.data) * grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        axes = _normalize_axis(axis, self.data.ndim)

        if axes is None:
            denom = self.data.size
        else:
            denom = np.prod([self.data.shape[a] for a in axes])

        return self.sum(axis=axis, keepdims=keepdims) / denom

    def max(self, axis=None, keepdims=False):
        values = self.data.max(axis=axis, keepdims=keepdims)
        out = Tensor(
            values,
            self.requires_grad,
            _children=(self,),
            _op="max"
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad
                max_values = values
                axes = _normalize_axis(axis, self.data.ndim)

                if axes is not None and not keepdims:
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, ax)
                        max_values = np.expand_dims(max_values, ax)

                mask = self.data == max_values
                count = mask.sum(axis=axis, keepdims=True) if axis is not None else mask.sum()
                self.grad += mask * grad / count

        out._backward = _backward
        return out

    def exp(self):
        out_data = np.exp(self.data)
        out = Tensor(
            out_data,
            self.requires_grad,
            _children=(self,),
            _op="exp"
        )

        def _backward():
            if self.requires_grad:
                self.grad += out_data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(
            np.log(self.data),
            self.requires_grad,
            _children=(self,),
            _op="log"
        )

        def _backward():
            if self.requires_grad:
                self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out

    def abs(self):
        out = Tensor(
            np.abs(self.data),
            self.requires_grad,
            _children=(self,),
            _op="abs"
        )

        def _backward():
            if self.requires_grad:
                self.grad += np.sign(self.data) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(
            np.maximum(0, self.data),
            self.requires_grad,
            _children=(self,),
            _op="relu"
        )

        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        sig = 1 / (1 + np.exp(-self.data))
        out = Tensor(
            sig,
            self.requires_grad,
            _children=(self,),
            _op="sigmoid"
        )

        def _backward():
            if self.requires_grad:
                self.grad += sig * (1 - sig) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        tanh_value = np.tanh(self.data)
        out = Tensor(
            tanh_value,
            self.requires_grad,
            _children=(self,),
            _op="tanh"
        )

        def _backward():
            if self.requires_grad:
                self.grad += (1 - tanh_value ** 2) * out.grad

        out._backward = _backward
        return out

    def log_softmax(self, axis=-1):
        shifted = self - Tensor(self.data.max(axis=axis, keepdims=True))
        return shifted - shifted.exp().sum(axis=axis, keepdims=True).log()

    def softmax(self, axis=-1):
        return self.log_softmax(axis=axis).exp()

    def __getitem__(self, index):
        out = Tensor(
            self.data[index],
            self.requires_grad,
            _children=(self,),
            _op="getitem"
        )

        def _backward():
            if self.requires_grad:
                try:
                    np.add.at(self.grad, index, out.grad)
                except Exception:
                    grad_holder = np.zeros_like(self.data)
                    grad_holder[index] += out.grad
                    self.grad += grad_holder

        out._backward = _backward
        return out

    def __repr__(self):
        return (
            f"Tensor(shape={self.data.shape}, "
            f"requires_grad={self.requires_grad})\n"
            f"Data:\n{self.data}"
        )


class Parameter(Tensor):
    def __init__(self, data_or_shape, init="normal", requires_grad=True, name=None):
        if isinstance(data_or_shape, tuple):
            if init == "zeros":
                data = np.zeros(data_or_shape)
            elif init == "ones":
                data = np.ones(data_or_shape)
            elif init == "xavier":
                fan_in = data_or_shape[0]
                fan_out = data_or_shape[1] if len(data_or_shape) > 1 else fan_in
                limit = np.sqrt(6 / (fan_in + fan_out))
                data = np.random.uniform(-limit, limit, data_or_shape)
            elif init == "he":
                fan_in = data_or_shape[0]
                data = np.random.randn(*data_or_shape) * np.sqrt(2 / fan_in)
            else:
                data = np.random.randn(*data_or_shape) * 0.01
        else:
            data = data_or_shape

        super().__init__(data, requires_grad=requires_grad, name=name)

    def __repr__(self):
        return (
            f"Parameter(shape={self.data.shape}, "
            f"requires_grad={self.requires_grad})\n"
            f"Data:\n{self.data}"
        )
