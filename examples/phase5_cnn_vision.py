import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ai


ai.manual_seed(42)
np.random.seed(42)

X = np.random.randn(64, 1, 8, 8)

center_signal = X[:, :, 2:6, 2:6].sum(axis=(1, 2, 3))
y = (center_signal > 0).astype(np.int64)

dataset = ai.data.TensorDataset(X, y)
train_dataset, val_dataset = ai.data.random_split(dataset, [48, 16], seed=42)

train_loader = ai.data.DataLoader(train_dataset, batch_size=8, shuffle=True, seed=42)
val_loader = ai.data.DataLoader(val_dataset, batch_size=8, shuffle=False)

model = ai.nn.Sequential(
    ai.vision.Conv2d(1, 4, kernel_size=3, padding=1),
    ai.nn.ReLU(),
    ai.vision.MaxPool2d(kernel_size=2),
    ai.vision.Conv2d(4, 8, kernel_size=3, padding=1),
    ai.nn.ReLU(),
    ai.vision.GlobalAvgPool2d(),
    ai.nn.Flatten(),
    ai.nn.Linear(8, 2)
)

loss_fn = ai.losses.CrossEntropyLoss()
optimizer = ai.optim.Adam(model.parameters(), lr=0.02)

trainer = ai.training.Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    metrics=[ai.training.accuracy]
)

trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20,
    log_every=5
)

results = trainer.evaluate(val_loader)

print("\nFinal validation results:")
print(results)
