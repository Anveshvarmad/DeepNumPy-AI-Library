import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ai


ai.manual_seed(42)

model = ai.nn.Sequential(
    ai.nn.Linear(2, 8),
    ai.nn.ReLU(),
    ai.nn.Linear(8, 1),
    ai.nn.Sigmoid()
)

X = ai.Tensor(
    np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]),
    requires_grad=False,
    name="X"
)

y = ai.Tensor(
    np.array([
        [0.0],
        [1.0],
        [1.0],
        [0.0],
    ]),
    requires_grad=False,
    name="y"
)

predictions = model(X)
loss = ai.losses.BCELoss()(predictions, y)

loss.name = "binary_cross_entropy_loss"

print("Model Summary:")
print(ai.utils.model_summary(model))

print("\nLoss:")
print(loss)

loss.backward()

graph_path = ai.utils.save_graph(
    loss,
    filename="docs/phase8_xor_computation_graph",
    graph_name="XOR Computation Graph"
)

print(f"\nSaved computation graph DOT file to: {graph_path}")
print("You can paste this DOT file into any Graphviz viewer.")
