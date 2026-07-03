import numpy as np
import ai


def test_conv2d_forward_shape():
    ai.manual_seed(1)

    x = ai.Tensor(np.random.randn(2, 3, 8, 8), requires_grad=False)
    conv = ai.vision.Conv2d(3, 6, kernel_size=3, padding=1)

    y = conv(x)

    assert y.shape == (2, 6, 8, 8)


def test_conv2d_backward_shapes():
    ai.manual_seed(2)

    x = ai.Tensor(np.random.randn(2, 3, 5, 5), requires_grad=True)
    conv = ai.vision.Conv2d(3, 4, kernel_size=3, padding=1)

    y = conv(x)
    loss = y.mean()
    loss.backward()

    assert x.grad.shape == x.shape
    assert conv.weight.grad.shape == conv.weight.shape
    assert conv.bias.grad.shape == conv.bias.shape


def test_maxpool2d_forward_shape():
    x = ai.Tensor(np.random.randn(2, 3, 8, 8), requires_grad=False)
    pool = ai.vision.MaxPool2d(kernel_size=2)

    y = pool(x)

    assert y.shape == (2, 3, 4, 4)


def test_maxpool2d_backward_shape():
    x = ai.Tensor(np.random.randn(1, 1, 4, 4), requires_grad=True)
    pool = ai.vision.MaxPool2d(kernel_size=2)

    y = pool(x)
    loss = y.sum()
    loss.backward()

    assert x.grad.shape == x.shape


def test_avgpool2d_backward_values():
    x = ai.Tensor(np.ones((1, 1, 2, 2)), requires_grad=True)
    pool = ai.vision.AvgPool2d(kernel_size=2)

    y = pool(x)
    y.backward(np.ones_like(y.data))

    expected = np.ones((1, 1, 2, 2)) * 0.25

    assert np.allclose(x.grad, expected)


def test_global_avg_pool_shape():
    x = ai.Tensor(np.random.randn(4, 8, 5, 5), requires_grad=False)
    pool = ai.vision.GlobalAvgPool2d()

    y = pool(x)

    assert y.shape == (4, 8, 1, 1)


def test_cnn_model_backward():
    ai.manual_seed(3)

    x = ai.Tensor(np.random.randn(2, 1, 8, 8), requires_grad=False)

    model = ai.nn.Sequential(
        ai.vision.Conv2d(1, 4, kernel_size=3, padding=1),
        ai.nn.ReLU(),
        ai.vision.MaxPool2d(kernel_size=2),
        ai.vision.GlobalAvgPool2d(),
        ai.nn.Flatten(),
        ai.nn.Linear(4, 2)
    )

    y = model(x)
    loss = y.mean()
    loss.backward()

    params = model.parameters()

    assert len(params) == 4

    for parameter in params:
        assert parameter.grad.shape == parameter.shape
