import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ai


ai.manual_seed(42)

x = ai.Tensor([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
], requires_grad=False)

model = ai.nn.Sequential(
    ai.nn.Linear(2, 8),
    ai.nn.BatchNorm1d(8),
    ai.nn.ReLU(),
    ai.nn.Dropout(p=0.2),
    ai.nn.Linear(8, 4),
    ai.nn.LayerNorm(4),
    ai.nn.Tanh(),
    ai.nn.Linear(4, 1)
)

print(model)

out = model(x)
loss = (out ** 2).mean()

print("\nOutput:")
print(out)

print("\nLoss:")
print(loss)

loss.backward()

print("\nNumber of parameters:")
print(len(model.parameters()))

print("\nParameter shapes:")
for parameter in model.parameters():
    print(parameter.shape)

print("\nFirst layer weight gradient shape:")
print(model[0].weight.grad.shape)
