import numpy as np
import ai


def test_add_backward():
    x = ai.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x + 2).sum()
    y.backward()

    assert np.allclose(x.grad, np.array([1.0, 1.0, 1.0]))


def test_mul_backward():
    x = ai.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x * x).sum()
    y.backward()

    assert np.allclose(x.grad, np.array([2.0, 4.0, 6.0]))


def test_matmul_backward():
    x = ai.Tensor([[1.0, 2.0]], requires_grad=True)
    W = ai.Tensor([[3.0], [4.0]], requires_grad=True)

    y = x @ W
    y.backward()

    assert np.allclose(x.grad, np.array([[3.0, 4.0]]))
    assert np.allclose(W.grad, np.array([[1.0], [2.0]]))


def test_relu_backward():
    x = ai.Tensor([-1.0, 0.0, 2.0], requires_grad=True)
    y = x.relu().sum()
    y.backward()

    assert np.allclose(x.grad, np.array([0.0, 0.0, 1.0]))


def test_mean_backward():
    x = ai.Tensor([2.0, 4.0], requires_grad=True)
    y = x.mean()
    y.backward()

    assert np.allclose(x.grad, np.array([0.5, 0.5]))
