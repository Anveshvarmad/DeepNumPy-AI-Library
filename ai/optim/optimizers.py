import numpy as np


def clip_grad_norm(parameters, max_norm, eps=1e-6):
    parameters = list(parameters)

    total_norm = 0.0

    for parameter in parameters:
        if parameter.grad is not None:
            total_norm += float((parameter.grad ** 2).sum())

    total_norm = total_norm ** 0.5

    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)

        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad *= scale

    return total_norm


class Optimizer:
    def __init__(self, parameters, lr=0.01, weight_decay=0.0, max_grad_norm=None):
        self.parameters = list(parameters)
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()

    def _grad(self, parameter):
        if parameter.grad is None:
            return None

        grad = parameter.grad

        if self.weight_decay != 0:
            grad = grad + self.weight_decay * parameter.data

        return grad

    def _clip_if_needed(self):
        if self.max_grad_norm is not None:
            clip_grad_norm(self.parameters, self.max_grad_norm)


class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0, max_grad_norm=None):
        super().__init__(parameters, lr, weight_decay, max_grad_norm)
        self.momentum = momentum
        self.velocity = [np.zeros_like(parameter.data) for parameter in self.parameters]

    def step(self):
        self._clip_if_needed()

        for index, parameter in enumerate(self.parameters):
            grad = self._grad(parameter)

            if grad is None:
                continue

            if self.momentum != 0:
                self.velocity[index] = self.momentum * self.velocity[index] + grad
                update = self.velocity[index]
            else:
                update = grad

            parameter.data -= self.lr * update


class Adam(Optimizer):
    def __init__(
        self,
        parameters,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
        max_grad_norm=None
    ):
        super().__init__(parameters, lr, weight_decay, max_grad_norm)

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.m = [np.zeros_like(parameter.data) for parameter in self.parameters]
        self.v = [np.zeros_like(parameter.data) for parameter in self.parameters]

    def step(self):
        self._clip_if_needed()
        self.t += 1

        for index, parameter in enumerate(self.parameters):
            grad = self._grad(parameter)

            if grad is None:
                continue

            self.m[index] = self.beta1 * self.m[index] + (1 - self.beta1) * grad
            self.v[index] = self.beta2 * self.v[index] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[index] / (1 - self.beta1 ** self.t)
            v_hat = self.v[index] / (1 - self.beta2 ** self.t)

            parameter.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class RMSProp(Optimizer):
    def __init__(
        self,
        parameters,
        lr=0.001,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0.0,
        max_grad_norm=None
    ):
        super().__init__(parameters, lr, weight_decay, max_grad_norm)

        self.alpha = alpha
        self.eps = eps
        self.square_avg = [np.zeros_like(parameter.data) for parameter in self.parameters]

    def step(self):
        self._clip_if_needed()

        for index, parameter in enumerate(self.parameters):
            grad = self._grad(parameter)

            if grad is None:
                continue

            self.square_avg[index] = (
                self.alpha * self.square_avg[index]
                + (1 - self.alpha) * (grad ** 2)
            )

            parameter.data -= self.lr * grad / (np.sqrt(self.square_avg[index]) + self.eps)


class Adagrad(Optimizer):
    def __init__(
        self,
        parameters,
        lr=0.01,
        eps=1e-8,
        weight_decay=0.0,
        max_grad_norm=None
    ):
        super().__init__(parameters, lr, weight_decay, max_grad_norm)

        self.eps = eps
        self.sum_squares = [np.zeros_like(parameter.data) for parameter in self.parameters]

    def step(self):
        self._clip_if_needed()

        for index, parameter in enumerate(self.parameters):
            grad = self._grad(parameter)

            if grad is None:
                continue

            self.sum_squares[index] += grad ** 2
            parameter.data -= self.lr * grad / (np.sqrt(self.sum_squares[index]) + self.eps)
