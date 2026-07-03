import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ai


ai.manual_seed(42)

X = ai.Tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])

y = ai.Tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0],
])

model = ai.nn.Sequential(
    ai.nn.Linear(2, 8),
    ai.nn.Tanh(),
    ai.nn.Linear(8, 1),
    ai.nn.Sigmoid()
)

loss_fn = ai.losses.BCELoss()
optimizer = ai.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(3000):
    predictions = model(X)
    loss = loss_fn(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"epoch={epoch}, loss={loss.item():.6f}")

print("\nFinal predictions:")
print(model(X).data.round(3))
