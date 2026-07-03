import os
import numpy as np
import ai


def test_tensor_dataset_and_dataloader_shapes():
    X = np.ones((10, 3))
    y = np.ones((10,))

    dataset = ai.data.TensorDataset(X, y)
    loader = ai.data.DataLoader(dataset, batch_size=4, shuffle=False)

    batches = list(loader)

    assert len(batches) == 3
    assert batches[0][0].shape == (4, 3)
    assert batches[0][1].shape == (4,)


def test_random_split_lengths():
    X = np.ones((10, 3))
    y = np.ones((10,))

    dataset = ai.data.TensorDataset(X, y)
    train_dataset, val_dataset = ai.data.random_split(dataset, [7, 3], seed=1)

    assert len(train_dataset) == 7
    assert len(val_dataset) == 3


def test_accuracy_metric():
    prediction = ai.Tensor([
        [2.0, 0.1],
        [0.2, 1.5],
        [3.0, 0.1]
    ])

    target = ai.Tensor([0, 1, 0])

    result = ai.training.accuracy(prediction, target)

    assert result == 1.0


def test_trainer_reduces_loss():
    ai.manual_seed(7)

    X = np.linspace(-1, 1, 40).reshape(-1, 1)
    y = 2 * X + 1

    dataset = ai.data.TensorDataset(X, y)
    loader = ai.data.DataLoader(dataset, batch_size=10, shuffle=True, seed=7)

    model = ai.nn.Sequential(
        ai.nn.Linear(1, 1)
    )

    loss_fn = ai.losses.MSELoss()
    optimizer = ai.optim.Adam(model.parameters(), lr=0.05)

    trainer = ai.training.Trainer(model, loss_fn, optimizer)

    before = trainer.evaluate(loader)["loss"]
    trainer.fit(loader, epochs=80, log_every=None)
    after = trainer.evaluate(loader)["loss"]

    assert after < before


def test_checkpoint_save_and_load(tmp_path):
    ai.manual_seed(9)

    model = ai.nn.Sequential(
        ai.nn.Linear(2, 4),
        ai.nn.ReLU(),
        ai.nn.Linear(4, 1)
    )

    path = tmp_path / "checkpoint.npz"

    ai.training.save_checkpoint(
        path,
        model,
        epoch=3,
        history={"train_loss": [1.0, 0.5]},
        extra={"name": "test"}
    )

    new_model = ai.nn.Sequential(
        ai.nn.Linear(2, 4),
        ai.nn.ReLU(),
        ai.nn.Linear(4, 1)
    )

    metadata = ai.training.load_checkpoint(path, new_model)

    assert metadata["epoch"] == 3
    assert metadata["extra"]["name"] == "test"

    for old_param, new_param in zip(model.parameters(), new_model.parameters()):
        assert np.allclose(old_param.data, new_param.data)
