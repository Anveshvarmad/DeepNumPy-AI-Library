import numpy as np

from ..core import Tensor, Parameter, stack
from ..nn.modules import Module


def _zeros(batch_size, hidden_size):
    return Tensor(np.zeros((batch_size, hidden_size)), requires_grad=False)


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = Parameter(
            np.random.randn(num_embeddings, embedding_dim) * 0.01,
            name="weight"
        )

    def forward(self, x):
        indices = x.data.astype(np.int64)

        if indices.min() < 0 or indices.max() >= self.num_embeddings:
            raise ValueError("Embedding input contains index outside vocabulary range")

        out_data = self.weight.data[indices]

        out = Tensor(
            out_data,
            requires_grad=self.weight.requires_grad,
            _children=(self.weight,),
            _op="embedding"
        )

        def _backward():
            if self.weight.requires_grad:
                np.add.at(self.weight.grad, indices, out.grad)

        out._backward = _backward
        return out

    def __repr__(self):
        return (
            f"Embedding(num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim})"
        )


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = Parameter((input_size, hidden_size), init="xavier", name="weight_ih")
        self.weight_hh = Parameter((hidden_size, hidden_size), init="xavier", name="weight_hh")

        self.bias = Parameter((1, hidden_size), init="zeros", name="bias") if bias else None

    def forward(self, x, hidden):
        out = x @ self.weight_ih + hidden @ self.weight_hh

        if self.bias is not None:
            out = out + self.bias

        return out.tanh()

    def __repr__(self):
        return (
            f"RNNCell(input_size={self.input_size}, "
            f"hidden_size={self.hidden_size}, "
            f"bias={self.bias is not None})"
        )


class RNN(Module):
    def __init__(self, input_size, hidden_size, return_sequences=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences

        self.cell = RNNCell(input_size, hidden_size)

    def forward(self, x, hidden=None):
        if x.data.ndim != 3:
            raise ValueError("RNN expects input shape (batch, sequence_length, input_size)")

        batch_size, sequence_length, input_size = x.shape

        if input_size != self.input_size:
            raise ValueError(f"Expected input_size={self.input_size}, got {input_size}")

        if hidden is None:
            hidden = _zeros(batch_size, self.hidden_size)

        outputs = []

        for timestep in range(sequence_length):
            xt = x[:, timestep, :]
            hidden = self.cell(xt, hidden)
            outputs.append(hidden)

        output_sequence = stack(outputs, axis=1)

        if self.return_sequences:
            return output_sequence, hidden

        return hidden

    def __repr__(self):
        return (
            f"RNN(input_size={self.input_size}, "
            f"hidden_size={self.hidden_size}, "
            f"return_sequences={self.return_sequences})"
        )


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = Parameter((input_size, 4 * hidden_size), init="xavier", name="weight_ih")
        self.weight_hh = Parameter((hidden_size, 4 * hidden_size), init="xavier", name="weight_hh")

        self.bias = Parameter((1, 4 * hidden_size), init="zeros", name="bias") if bias else None

    def forward(self, x, hidden, cell):
        gates = x @ self.weight_ih + hidden @ self.weight_hh

        if self.bias is not None:
            gates = gates + self.bias

        h = self.hidden_size

        input_gate = gates[:, 0:h].sigmoid()
        forget_gate = gates[:, h:2 * h].sigmoid()
        candidate = gates[:, 2 * h:3 * h].tanh()
        output_gate = gates[:, 3 * h:4 * h].sigmoid()

        next_cell = forget_gate * cell + input_gate * candidate
        next_hidden = output_gate * next_cell.tanh()

        return next_hidden, next_cell

    def __repr__(self):
        return (
            f"LSTMCell(input_size={self.input_size}, "
            f"hidden_size={self.hidden_size}, "
            f"bias={self.bias is not None})"
        )


class LSTM(Module):
    def __init__(self, input_size, hidden_size, return_sequences=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences

        self.cell = LSTMCell(input_size, hidden_size)

    def forward(self, x, hidden=None, cell=None):
        if x.data.ndim != 3:
            raise ValueError("LSTM expects input shape (batch, sequence_length, input_size)")

        batch_size, sequence_length, input_size = x.shape

        if input_size != self.input_size:
            raise ValueError(f"Expected input_size={self.input_size}, got {input_size}")

        if hidden is None:
            hidden = _zeros(batch_size, self.hidden_size)

        if cell is None:
            cell = _zeros(batch_size, self.hidden_size)

        outputs = []

        for timestep in range(sequence_length):
            xt = x[:, timestep, :]
            hidden, cell = self.cell(xt, hidden, cell)
            outputs.append(hidden)

        output_sequence = stack(outputs, axis=1)

        if self.return_sequences:
            return output_sequence, hidden, cell

        return hidden

    def __repr__(self):
        return (
            f"LSTM(input_size={self.input_size}, "
            f"hidden_size={self.hidden_size}, "
            f"return_sequences={self.return_sequences})"
        )
