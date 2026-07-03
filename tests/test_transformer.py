import numpy as np
import ai


def test_tensor_permute_backward():
    x = ai.Tensor(np.random.randn(2, 3, 4), requires_grad=True)

    y = x.permute(0, 2, 1)
    loss = y.sum()
    loss.backward()

    assert y.shape == (2, 4, 3)
    assert x.grad.shape == x.shape
    assert np.allclose(x.grad, np.ones_like(x.data))


def test_positional_encoding_shape():
    pe = ai.transformer.PositionalEncoding(d_model=8, max_len=20)

    x = ai.Tensor(np.zeros((4, 10, 8)), requires_grad=True)
    y = pe(x)

    assert y.shape == (4, 10, 8)


def test_scaled_dot_product_attention_shape():
    attention = ai.transformer.ScaledDotProductAttention()

    q = ai.Tensor(np.random.randn(2, 4, 5, 8), requires_grad=True)
    k = ai.Tensor(np.random.randn(2, 4, 5, 8), requires_grad=True)
    v = ai.Tensor(np.random.randn(2, 4, 5, 8), requires_grad=True)

    out, weights = attention(q, k, v, return_attention=True)

    assert out.shape == (2, 4, 5, 8)
    assert weights.shape == (2, 4, 5, 5)


def test_attention_backward():
    attention = ai.transformer.ScaledDotProductAttention()

    q = ai.Tensor(np.random.randn(2, 2, 3, 4), requires_grad=True)
    k = ai.Tensor(np.random.randn(2, 2, 3, 4), requires_grad=True)
    v = ai.Tensor(np.random.randn(2, 2, 3, 4), requires_grad=True)

    out = attention(q, k, v)
    loss = out.mean()
    loss.backward()

    assert q.grad.shape == q.shape
    assert k.grad.shape == k.shape
    assert v.grad.shape == v.shape


def test_multi_head_attention_shape():
    mha = ai.transformer.MultiHeadAttention(embed_dim=16, num_heads=4)

    x = ai.Tensor(np.random.randn(3, 6, 16), requires_grad=True)

    y = mha(x)

    assert y.shape == (3, 6, 16)


def test_multi_head_attention_backward():
    mha = ai.transformer.MultiHeadAttention(embed_dim=16, num_heads=4)

    x = ai.Tensor(np.random.randn(3, 6, 16), requires_grad=True)

    y = mha(x)
    loss = y.mean()
    loss.backward()

    assert x.grad.shape == x.shape

    for parameter in mha.parameters():
        assert parameter.grad.shape == parameter.shape


def test_feed_forward_shape():
    ff = ai.transformer.FeedForward(embed_dim=16, hidden_dim=32)

    x = ai.Tensor(np.random.randn(3, 6, 16), requires_grad=True)
    y = ff(x)

    assert y.shape == (3, 6, 16)


def test_transformer_encoder_block_shape():
    block = ai.transformer.TransformerEncoderBlock(
        embed_dim=16,
        num_heads=4,
        ff_hidden_dim=32
    )

    x = ai.Tensor(np.random.randn(3, 6, 16), requires_grad=True)

    y = block(x)

    assert y.shape == (3, 6, 16)


def test_transformer_encoder_block_backward():
    block = ai.transformer.TransformerEncoderBlock(
        embed_dim=16,
        num_heads=4,
        ff_hidden_dim=32
    )

    x = ai.Tensor(np.random.randn(3, 6, 16), requires_grad=True)

    y = block(x)
    loss = y.mean()
    loss.backward()

    assert x.grad.shape == x.shape

    for parameter in block.parameters():
        assert parameter.grad.shape == parameter.shape


def test_transformer_encoder_shape():
    encoder = ai.transformer.TransformerEncoder(
        num_layers=2,
        embed_dim=16,
        num_heads=4,
        ff_hidden_dim=32
    )

    x = ai.Tensor(np.random.randn(3, 6, 16), requires_grad=True)

    y = encoder(x)

    assert y.shape == (3, 6, 16)


def test_transformer_classifier_backward():
    model = ai.transformer.TransformerClassifier(
        vocab_size=20,
        embed_dim=16,
        num_heads=4,
        ff_hidden_dim=32,
        num_classes=2,
        num_layers=1,
        max_len=8
    )

    x = ai.Tensor(np.random.randint(0, 20, size=(4, 8)))
    target = np.array([0, 1, 0, 1])

    logits = model(x)
    loss = ai.losses.CrossEntropyLoss()(logits, target)
    loss.backward()

    assert logits.shape == (4, 2)

    for parameter in model.parameters():
        assert parameter.grad.shape == parameter.shape
