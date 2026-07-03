import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ai


ai.manual_seed(42)
np.random.seed(42)


class SequenceClassifier(ai.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super().__init__()

        self.embedding = ai.sequence.Embedding(vocab_size, embedding_dim)
        self.lstm = ai.sequence.LSTM(embedding_dim, hidden_size, return_sequences=False)
        self.fc = ai.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        hidden = self.lstm(embedded)
        return self.fc(hidden)


vocab_size = 20
sequence_length = 6
num_samples = 240

X = np.random.randint(0, vocab_size, size=(num_samples, sequence_length))

signal = X[:, :3].sum(axis=1) - X[:, 3:].sum(axis=1)
y = (signal > 0).astype(np.int64)

dataset = ai.data.TensorDataset(X, y)
train_dataset, val_dataset = ai.data.random_split(dataset, [200, 40], seed=42)

train_loader = ai.data.DataLoader(train_dataset, batch_size=32, shuffle=True, seed=42)
val_loader = ai.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

model = SequenceClassifier(
    vocab_size=vocab_size,
    embedding_dim=12,
    hidden_size=16,
    num_classes=2
)

loss_fn = ai.losses.CrossEntropyLoss()
optimizer = ai.optim.Adam(model.parameters(), lr=0.03, max_grad_norm=5.0)

trainer = ai.training.Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    metrics=[ai.training.accuracy]
)

trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=25,
    log_every=5
)

results = trainer.evaluate(val_loader)

print("\nFinal validation results:")
print(results)
