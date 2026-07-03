import numpy as np
import ai


def test_mse_loss_backward():
    prediction = ai.Tensor([[1.0], [2.0]], requires_grad=True)
    target = ai.Tensor([[0.0], [0.0]])

    loss_fn = ai.losses.MSELoss()
    loss = loss_fn(prediction, target)

    loss.backward()

    assert np.allclose(prediction.grad, np.array([[1.0], [2.0]]))


def test_bce_loss_backward_shape():
    prediction = ai.Tensor([[0.8], [0.2]], requires_grad=True)
    target = ai.Tensor([[1.0], [0.0]])

    loss_fn = ai.losses.BCELoss()
    loss = loss_fn(prediction, target)

    loss.backward()

    assert prediction.grad.shape == prediction.shape


def test_cross_entropy_loss_backward_shape():
    logits = ai.Tensor([
        [2.0, 1.0, 0.1],
        [0.2, 0.5, 2.0]
    ], requires_grad=True)

    target = np.array([0, 2])

    loss_fn = ai.losses.CrossEntropyLoss()
    loss = loss_fn(logits, target)

    loss.backward()

    assert logits.grad.shape == logits.shape


def test_sgd_step_updates_parameter():
    parameter = ai.Parameter(np.array([1.0]))

    parameter.grad = np.array([2.0])

    optimizer = ai.optim.SGD([parameter], lr=0.1)
    optimizer.step()

    assert np.allclose(parameter.data, np.array([0.8]))


def test_adam_step_updates_parameter():
    parameter = ai.Parameter(np.array([1.0]))

    parameter.grad = np.array([2.0])

    optimizer = ai.optim.Adam([parameter], lr=0.1)
    optimizer.step()

    assert parameter.data[0] < 1.0


def test_clip_grad_norm():
    parameter = ai.Parameter(np.array([1.0, 2.0]))

    parameter.grad = np.array([3.0, 4.0])

    norm = ai.optim.clip_grad_norm([parameter], max_norm=1.0)

    assert np.isclose(norm, 5.0)
    assert np.linalg.norm(parameter.grad) <= 1.000001
