import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ai


ai.manual_seed(42)
np.random.seed(42)

class_0 = np.random.randn(100, 2) * 0.5 + np.array([-1.5, -1.5])
class_1 = np.random.randn(100, 2) * 0.5 + np.array([1.5, 1.5])

X = np.vstack([class_0, class_1])
y = np.array([0] * 100 + [1] * 100)

dataset = ai.data.TensorDataset(X, y)
train_dataset, val_dataset = ai.data.random_split(dataset, [160, 40], seed=42)

train_loader = ai.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    seed=42
)

val_loader = ai.data.DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)

model = ai.nn.Sequential(
    ai.nn.Linear(2, 16),
    ai.nn.ReLU(),
    ai.nn.Linear(16, 2)
)

loss_fn = ai.losses.CrossEntropyLoss()
optimizer = ai.optim.Adam(model.parameters(), lr=0.03)

trainer = ai.training.Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    metrics=[ai.training.accuracy]
)

history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=25,
    log_every=5
)

ai.training.save_checkpoint(
    "phase4_classifier_checkpoint.npz",
    model=model,
    epoch=25,
    history=history,
    extra={"example": "phase4_training_workflow"}
)

print("\nSaved checkpoint: phase4_classifier_checkpoint.npz")

new_model = ai.nn.Sequential(
    ai.nn.Linear(2, 16),
    ai.nn.ReLU(),
    ai.nn.Linear(16, 2)
)

metadata = ai.training.load_checkpoint(
    "phase4_classifier_checkpoint.npz",
    new_model
)

print("Loaded checkpoint metadata:")
print(metadata)

new_trainer = ai.training.Trainer(
    model=new_model,
    loss_fn=loss_fn,
    optimizer=ai.optim.Adam(new_model.parameters(), lr=0.03),
    metrics=[ai.training.accuracy]
)

results = new_trainer.evaluate(val_loader)

print("\nReloaded model validation results:")
print(results)
