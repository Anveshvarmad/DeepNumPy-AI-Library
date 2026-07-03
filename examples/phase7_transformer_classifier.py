import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ai


ai.manual_seed(42)
np.random.seed(42)

vocab_size = 30
sequence_length = 8
num_samples = 240

X = np.random.randint(0, vocab_size, size=(num_samples, sequence_length))

left_score = X[:, :4].sum(axis=1)
right_score = X[:, 4:].sum(axis=1)

y = (left_score > right_score).astype(np.int64)

dataset = ai.data.TensorDataset(X, y)
train_dataset, val_dataset = ai.data.random_split(dataset, [200, 40], seed=42)

train_loader = ai.data.DataLoader(train_dataset, batch_size=32, shuffle=True, seed=42)
val_loader = ai.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

model = ai.transformer.TransformerClassifier(
    vocab_size=vocab_size,
    embed_dim=16,
    num_heads=4,
    ff_hidden_dim=32,
    num_classes=2,
    num_layers=1,
    max_len=sequence_length,
    dropout=0.1
)

loss_fn = ai.losses.CrossEntropyLoss()
optimizer = ai.optim.Adam(model.parameters(), lr=0.02, max_grad_norm=5.0)

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
