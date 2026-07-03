import numpy as np
import ai


def test_embedding_forward_shape():
    embedding = ai.sequence.Embedding(num_embeddings=10, embedding_dim=4)

    x = ai.Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    y = embedding(x)

    assert y.shape == (2, 3, 4)


def test_embedding_backward_repeated_indices():
    embedding = ai.sequence.Embedding(num_embeddings=5, embedding_dim=3)

    x = ai.Tensor(np.array([[1, 1, 2]]))
    y = embedding(x)
    loss = y.sum()
    loss.backward()

    assert np.allclose(embedding.weight.grad[1], np.ones(3) * 2)
    assert np.allclose(embedding.weight.grad[2], np.ones(3))


def test_rnn_cell_forward_shape():
    cell = ai.sequence.RNNCell(input_size=4, hidden_size=6)

    x = ai.Tensor(np.random.randn(3, 4))
    h = ai.Tensor(np.zeros((3, 6)))

    next_h = cell(x, h)

    assert next_h.shape == (3, 6)


def test_rnn_forward_shapes():
    rnn = ai.sequence.RNN(input_size=4, hidden_size=6, return_sequences=True)

    x = ai.Tensor(np.random.randn(3, 5, 4), requires_grad=True)

    output_sequence, hidden = rnn(x)

    assert output_sequence.shape == (3, 5, 6)
    assert hidden.shape == (3, 6)


def test_rnn_backward():
    rnn = ai.sequence.RNN(input_size=4, hidden_size=6, return_sequences=False)

    x = ai.Tensor(np.random.randn(3, 5, 4), requires_grad=True)

    hidden = rnn(x)
    loss = hidden.mean()
    loss.backward()

    assert x.grad.shape == x.shape

    for parameter in rnn.parameters():
        assert parameter.grad.shape == parameter.shape


def test_lstm_cell_forward_shapes():
    cell = ai.sequence.LSTMCell(input_size=4, hidden_size=6)

    x = ai.Tensor(np.random.randn(3, 4))
    h = ai.Tensor(np.zeros((3, 6)))
    c = ai.Tensor(np.zeros((3, 6)))

    next_h, next_c = cell(x, h, c)

    assert next_h.shape == (3, 6)
    assert next_c.shape == (3, 6)


def test_lstm_forward_shapes():
    lstm = ai.sequence.LSTM(input_size=4, hidden_size=6, return_sequences=True)

    x = ai.Tensor(np.random.randn(3, 5, 4), requires_grad=True)

    output_sequence, hidden, cell = lstm(x)

    assert output_sequence.shape == (3, 5, 6)
    assert hidden.shape == (3, 6)
    assert cell.shape == (3, 6)


def test_lstm_backward():
    lstm = ai.sequence.LSTM(input_size=4, hidden_size=6, return_sequences=False)

    x = ai.Tensor(np.random.randn(3, 5, 4), requires_grad=True)

    hidden = lstm(x)
    loss = hidden.mean()
    loss.backward()

    assert x.grad.shape == x.shape

    for parameter in lstm.parameters():
        assert parameter.grad.shape == parameter.shape


def test_sequence_classifier_backward():
    class SequenceClassifier(ai.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = ai.sequence.Embedding(12, 5)
            self.rnn = ai.sequence.RNN(5, 7, return_sequences=False)
            self.fc = ai.nn.Linear(7, 2)

        def forward(self, x):
            embedded = self.embedding(x)
            hidden = self.rnn(embedded)
            return self.fc(hidden)

    model = SequenceClassifier()

    x = ai.Tensor(np.array([
        [1, 2, 3, 4],
        [4, 3, 2, 1]
    ]))

    target = np.array([0, 1])

    logits = model(x)
    loss = ai.losses.CrossEntropyLoss()(logits, target)

    loss.backward()

    assert logits.shape == (2, 2)

    for parameter in model.parameters():
        assert parameter.grad.shape == parameter.shape
