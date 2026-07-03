import math
import numpy as np

from ..core import Tensor
from ..nn.modules import Module, Linear, ReLU, Dropout, LayerNorm, ModuleList
from ..sequence import Embedding


class PositionalEncoding(Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len

        position = np.arange(max_len).reshape(max_len, 1)
        div_term = np.exp(
            np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        encoding = np.zeros((max_len, d_model))
        encoding[:, 0::2] = np.sin(position * div_term)

        if d_model > 1:
            encoding[:, 1::2] = np.cos(position * div_term[:encoding[:, 1::2].shape[1]])

        self.encoding = encoding.reshape(1, max_len, d_model)

    def forward(self, x):
        if x.data.ndim != 3:
            raise ValueError("PositionalEncoding expects input shape (batch, sequence_length, d_model)")

        sequence_length = x.shape[1]

        if sequence_length > self.max_len:
            raise ValueError("sequence length exceeds max_len")

        return x + Tensor(self.encoding[:, :sequence_length, :], requires_grad=False)

    def __repr__(self):
        return f"PositionalEncoding(d_model={self.d_model}, max_len={self.max_len})"


class ScaledDotProductAttention(Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = Dropout(dropout)

    def forward(self, query, key, value, mask=None, return_attention=False):
        d_k = query.shape[-1]

        scores = (query @ key.T) / math.sqrt(d_k)

        if mask is not None:
            if isinstance(mask, Tensor):
                mask_data = mask.data
            else:
                mask_data = np.array(mask)

            scores = scores + Tensor((1.0 - mask_data) * -1e9, requires_grad=False)

        attention = scores.softmax(axis=-1)
        attention = self.dropout(attention)

        output = attention @ value

        if return_attention:
            return output, attention

        return output

    def __repr__(self):
        return "ScaledDotProductAttention()"


class MultiHeadAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query_proj = Linear(embed_dim, embed_dim)
        self.key_proj = Linear(embed_dim, embed_dim)
        self.value_proj = Linear(embed_dim, embed_dim)
        self.output_proj = Linear(embed_dim, embed_dim)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def _split_heads(self, x):
        batch_size, sequence_length, _ = x.shape

        return x.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        batch_size, num_heads, sequence_length, head_dim = x.shape

        return x.permute(0, 2, 1, 3).reshape(
            batch_size,
            sequence_length,
            num_heads * head_dim
        )

    def forward(self, query, key=None, value=None, mask=None, return_attention=False):
        if key is None:
            key = query

        if value is None:
            value = key

        q = self._split_heads(self.query_proj(query))
        k = self._split_heads(self.key_proj(key))
        v = self._split_heads(self.value_proj(value))

        if return_attention:
            attention_output, attention_weights = self.attention(
                q,
                k,
                v,
                mask=mask,
                return_attention=True
            )
        else:
            attention_output = self.attention(q, k, v, mask=mask)
            attention_weights = None

        merged = self._merge_heads(attention_output)
        output = self.output_proj(merged)

        if return_attention:
            return output, attention_weights

        return output

    def __repr__(self):
        return (
            f"MultiHeadAttention(embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads})"
        )


class FeedForward(Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.linear1 = Linear(embed_dim, hidden_dim)
        self.activation = ReLU()
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x

    def __repr__(self):
        return (
            f"FeedForward(embed_dim={self.embed_dim}, "
            f"hidden_dim={self.hidden_dim})"
        )


class TransformerEncoderBlock(Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.0, eps=1e-5):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_hidden_dim = ff_hidden_dim

        self.self_attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.feed_forward = FeedForward(
            embed_dim=embed_dim,
            hidden_dim=ff_hidden_dim,
            dropout=dropout
        )

        self.norm1 = LayerNorm(embed_dim, eps=eps)
        self.norm2 = LayerNorm(embed_dim, eps=eps)
        self.dropout = Dropout(dropout)

    def forward(self, x, mask=None):
        attention_output = self.self_attention(x, mask=mask)
        x = self.norm1(x + self.dropout(attention_output))

        feed_forward_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(feed_forward_output))

        return x

    def __repr__(self):
        return (
            f"TransformerEncoderBlock(embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, "
            f"ff_hidden_dim={self.ff_hidden_dim})"
        )


class TransformerEncoder(Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_hidden_dim, dropout=0.0):
        super().__init__()

        self.num_layers = num_layers
        self.layers = ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_hidden_dim=ff_hidden_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)

        return x

    def __repr__(self):
        return (
            f"TransformerEncoder(num_layers={self.num_layers}, "
            f"layers={len(self.layers)})"
        )


class TransformerClassifier(Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        ff_hidden_dim,
        num_classes,
        num_layers=1,
        max_len=256,
        dropout=0.0
    ):
        super().__init__()

        self.embedding = Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=max_len)

        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            dropout=dropout
        )

        self.fc = Linear(embed_dim, num_classes)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x, mask=mask)

        pooled = x.mean(axis=1)

        return self.fc(pooled)

    def __repr__(self):
        return "TransformerClassifier()"
