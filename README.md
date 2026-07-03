# DeepNumPy AI Library

DeepNumPy AI Library is a PyTorch-like deep learning framework built from scratch using Python and NumPy.

I built this project to understand how deep learning libraries work internally. Instead of directly using PyTorch or TensorFlow, I implemented the core pieces myself, including tensors, reverse-mode automatic differentiation, neural network layers, optimizers, losses, data loaders, CNN layers, RNN/LSTM models, Transformer components, checkpointing, model summaries, and computation graph visualization.

The goal of this project is educational. It shows how a deep learning framework can be designed from first principles.

---

## What I Built

This project implements a small but complete deep learning library with an API similar to PyTorch.

Example:

```python
import ai

model = ai.nn.Sequential(
    ai.nn.Linear(2, 8),
    ai.nn.Tanh(),
    ai.nn.Linear(8, 1),
    ai.nn.Sigmoid()
)

loss_fn = ai.losses.BCELoss()
optimizer = ai.optim.Adam(model.parameters(), lr=0.05)
```

The library supports the full training flow:

```text
Tensor → Model → Loss → Backward Pass → Optimizer Step → Updated Parameters
```

---

## Why I Built This

Modern AI frameworks hide a lot of complexity behind simple APIs like:

```python
loss.backward()
optimizer.step()
```

I wanted to understand what actually happens behind those calls. So I implemented the core mechanics manually using NumPy.

This helped me understand:

* How tensors store data and gradients
* How computational graphs are created during the forward pass
* How reverse-mode automatic differentiation works
* How gradients flow backward through operations
* How neural network layers collect trainable parameters
* How optimizers update model weights
* How CNNs, RNNs, LSTMs, and Transformers are structured internally

---

## Features

### Core Autograd Engine

* Tensor class
* Parameter class
* Reverse-mode automatic differentiation
* Dynamic computational graph
* Broadcasting-aware gradients
* Matrix multiplication gradients
* Tensor operations:

  * addition
  * subtraction
  * multiplication
  * division
  * power
  * matrix multiplication
  * reshape
  * transpose
  * permute
  * sum
  * mean
  * max
  * slicing
  * stack

### Activation Functions

* ReLU
* Sigmoid
* Tanh
* Softmax
* LogSoftmax

### Neural Network API

* Module base class
* Sequential container
* ModuleList
* Linear layer
* Dropout
* Flatten
* LayerNorm
* BatchNorm1d
* Model parameter collection
* Save and load model state dictionaries

### Loss Functions

* MSELoss
* L1Loss
* BCELoss
* CrossEntropyLoss
* NLLLoss
* HuberLoss

### Optimizers

* SGD
* SGD with momentum
* Adam
* RMSProp
* Adagrad
* Weight decay
* Gradient clipping

### Data and Training Utilities

* Dataset
* TensorDataset
* DataLoader
* random_split
* Trainer class
* Accuracy metrics
* Binary accuracy
* Mean squared error
* Mean absolute error
* Checkpoint save/load

### Vision Layers

* Conv2d
* MaxPool2d
* AvgPool2d
* GlobalAvgPool2d

### Sequence Models

* Embedding
* RNNCell
* RNN
* LSTMCell
* LSTM

### Transformer Components

* PositionalEncoding
* ScaledDotProductAttention
* MultiHeadAttention
* FeedForward
* TransformerEncoderBlock
* TransformerEncoder
* TransformerClassifier

### Utilities

* Model summary
* Parameter counting
* Computation graph tracing
* DOT graph export for visualization

---

## Tech Stack

* Python
* NumPy
* Pytest
* Graphviz
* Matplotlib

---

## Architecture

The project is organized as a small deep learning framework.

```text
DeepNumPy-AI-Library/
├── ai/
│   ├── core/
│   │   └── tensor.py
│   ├── nn/
│   │   └── modules.py
│   ├── losses/
│   │   └── losses.py
│   ├── optim/
│   │   └── optimizers.py
│   ├── data/
│   │   ├── dataset.py
│   │   └── dataloader.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── metrics.py
│   │   └── checkpoint.py
│   ├── vision/
│   │   └── layers.py
│   ├── sequence/
│   │   └── layers.py
│   ├── transformer/
│   │   └── layers.py
│   └── utils/
│       ├── graph.py
│       └── summary.py
├── examples/
├── tests/
├── docs/
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## High-Level Architecture

```text
User Code
   |
   v
ai.nn / ai.vision / ai.sequence / ai.transformer
   |
   v
Tensor Operations
   |
   v
Dynamic Computational Graph
   |
   v
Reverse-Mode Autograd Engine
   |
   v
Gradients Stored in Parameters
   |
   v
Optimizer Updates Weights
```

---

## How the Code Works

### 1. Tensor and Parameter

The `Tensor` class is the core object of the library. It stores:

* actual data
* gradient
* whether gradients are required
* previous tensors in the computation graph
* backward function for each operation

A `Parameter` is a special type of tensor used for trainable model weights.

Example:

```python
x = ai.Tensor([[1.0, 2.0]], requires_grad=True)
W = ai.Parameter([[3.0], [4.0]])

y = x @ W
loss = y.mean()

loss.backward()
```

When `loss.backward()` is called, the library walks through the graph in reverse order and computes gradients for every tensor that requires gradients.

---

### 2. Computational Graph

Every tensor operation creates a new tensor and stores references to the tensors that created it.

For example:

```python
y = x @ W
```

creates a new tensor `y`, and internally stores that it came from `x` and `W`.

The graph is dynamic, which means it is built during the forward pass just like PyTorch.

---

### 3. Backpropagation

Backpropagation is implemented using reverse-mode automatic differentiation.

The process is:

1. Build a topological ordering of all tensors involved in the computation.
2. Start from the final loss.
3. Move backward through the graph.
4. Call each tensor’s stored backward function.
5. Accumulate gradients into trainable parameters.

This is the core idea behind `loss.backward()`.

---

### 4. Neural Network Modules

The `Module` class is the base class for layers and models.

It supports:

* `forward()`
* `parameters()`
* `train()`
* `eval()`
* `zero_grad()`
* `state_dict()`
* `load_state_dict()`

Example:

```python
model = ai.nn.Sequential(
    ai.nn.Linear(2, 8),
    ai.nn.ReLU(),
    ai.nn.Linear(8, 1)
)
```

The model automatically collects parameters from all child layers.

---

### 5. Loss Functions

Loss functions compare model predictions with target values.

Example:

```python
loss_fn = ai.losses.CrossEntropyLoss()
loss = loss_fn(logits, labels)
```

The loss is also a tensor, so calling `loss.backward()` computes gradients through the entire model.

---

### 6. Optimizers

Optimizers update model parameters using the gradients computed during backpropagation.

Example:

```python
optimizer = ai.optim.Adam(model.parameters(), lr=0.001)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

I implemented several optimizers, including SGD, Adam, RMSProp, and Adagrad.

---

### 7. Training Workflow

The training utilities make the library easier to use.

The `Trainer` class handles:

* training loop
* validation loop
* loss tracking
* metric tracking
* model train/eval mode switching

Example:

```python
trainer = ai.training.Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    metrics=[ai.training.accuracy]
)

history = trainer.fit(train_loader, epochs=10, val_loader=val_loader)
```

---

## Example: XOR Training

```python
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

print(model(X).data.round(3))
```

Expected output:

```text
[[0.]
 [1.]
 [1.]
 [0.]]
```

---

## Example: CNN Model

```python
import ai

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
```

This shows how convolutional neural network layers can be built using the same tensor and autograd engine.

---

## Example: LSTM Sequence Model

```python
import ai

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
```

This model uses an embedding layer, LSTM, and linear classifier.

---

## Example: Transformer Classifier

```python
import ai

model = ai.transformer.TransformerClassifier(
    vocab_size=1000,
    embed_dim=64,
    num_heads=4,
    ff_hidden_dim=128,
    num_classes=2,
    num_layers=2,
    max_len=128,
    dropout=0.1
)
```

The Transformer implementation includes self-attention, multi-head attention, feed-forward layers, residual connections, and layer normalization.

---

## Model Summary

```python
print(ai.utils.model_summary(model))
```

Example output:

```text
Layer                               Type                          Direct Params   Total Params    Trainable
----------------------------------------------------------------------------------------------------------
model                               Sequential                                0             33           33
  model.layers.0                    Linear                                  24             24           24
  model.layers.1                    ReLU                                     0              0            0
  model.layers.2                    Linear                                   9              9            9
----------------------------------------------------------------------------------------------------------
Total parameters: 33
Trainable parameters: 33
```

---

## Computation Graph Visualization

The library can export a computation graph as a DOT file.

```python
ai.utils.save_graph(
    loss,
    filename="docs/computation_graph",
    graph_name="Training Graph"
)
```

This creates:

```text
docs/computation_graph.dot
```

The DOT file can be viewed using Graphviz or online DOT viewers.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/DeepNumPy-AI-Library.git
cd DeepNumPy-AI-Library
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Run Examples

```bash
python examples/core_autograd_demo.py
python examples/phase3_xor_training.py
python examples/phase4_training_workflow.py
python examples/phase5_cnn_vision.py
python examples/phase6_sequence_models.py
python examples/phase7_transformer_classifier.py
python examples/phase8_graph_and_summary.py
```

---

## Run Tests

```bash
pytest
```

The project includes tests for:

* tensor operations
* autograd
* neural network layers
* losses
* optimizers
* training workflow
* CNN layers
* RNN/LSTM layers
* Transformer components
* graph utilities
* model summaries

