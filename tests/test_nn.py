import numpy as np
import ai


def test_linear_forward_shape():
    layer = ai.nn.Linear(3, 2)
    x = ai.Tensor(np.ones((4, 3)))

    y = layer(x)

    assert y.shape == (4, 2)


def test_linear_backward():
    layer = ai.nn.Linear(3, 2)
    x = ai.Tensor(np.ones((4, 3)))

    y = layer(x)
    loss = y.sum()
    loss.backward()

    assert layer.weight.grad.shape == (3, 2)
    assert layer.bias.grad.shape == (1, 2)


def test_sequential_parameters():
    model = ai.nn.Sequential(
        ai.nn.Linear(2, 4),
        ai.nn.ReLU(),
        ai.nn.Linear(4, 1)
    )

    params = model.parameters()

    assert len(params) == 4


def test_dropout_eval_mode():
    ai.manual_seed(1)

    layer = ai.nn.Dropout(p=0.5)
    layer.eval()

    x = ai.Tensor(np.ones((3, 3)))
    y = layer(x)

    assert np.allclose(x.data, y.data)


def test_layer_norm_shape():
    layer = ai.nn.LayerNorm(4)
    x = ai.Tensor(np.ones((2, 4)))

    y = layer(x)

    assert y.shape == (2, 4)


def test_batchnorm_shape():
    layer = ai.nn.BatchNorm1d(5)
    x = ai.Tensor(np.ones((8, 5)))

    y = layer(x)

    assert y.shape == (8, 5)
