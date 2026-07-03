import numpy as np

from ..core import Tensor, Parameter
from ..nn.modules import Module


def _pair(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

        kh, kw = self.kernel_size
        scale = np.sqrt(2.0 / (in_channels * kh * kw))

        self.weight = Parameter(
            np.random.randn(out_channels, in_channels, kh, kw) * scale,
            name="weight"
        )

        self.bias = Parameter(
            np.zeros((1, out_channels, 1, 1)),
            name="bias"
        ) if bias else None

    def forward(self, x):
        if x.data.ndim != 4:
            raise ValueError("Conv2d expects input shape (batch, channels, height, width)")

        n, c, h, w = x.data.shape
        out_channels, weight_channels, kh, kw = self.weight.data.shape

        if c != weight_channels:
            raise ValueError(f"Expected {weight_channels} channels, got {c}")

        sh, sw = self.stride
        ph, pw = self.padding

        x_padded = np.pad(
            x.data,
            ((0, 0), (0, 0), (ph, ph), (pw, pw)),
            mode="constant"
        )

        h_out = (h + 2 * ph - kh) // sh + 1
        w_out = (w + 2 * pw - kw) // sw + 1

        out_data = np.zeros((n, out_channels, h_out, w_out))

        for batch in range(n):
            for oc in range(out_channels):
                for i in range(h_out):
                    for j in range(w_out):
                        h_start = i * sh
                        w_start = j * sw
                        window = x_padded[
                            batch,
                            :,
                            h_start:h_start + kh,
                            w_start:w_start + kw
                        ]
                        out_data[batch, oc, i, j] = np.sum(window * self.weight.data[oc])

        if self.bias is not None:
            out_data += self.bias.data

        children = (x, self.weight) if self.bias is None else (x, self.weight, self.bias)

        out = Tensor(
            out_data,
            requires_grad=x.requires_grad or self.weight.requires_grad or (self.bias is not None and self.bias.requires_grad),
            _children=children,
            _op="conv2d"
        )

        def _backward():
            if out.grad is None:
                return

            dx_padded = np.zeros_like(x_padded) if x.requires_grad else None
            dw = np.zeros_like(self.weight.data) if self.weight.requires_grad else None
            db = np.zeros_like(self.bias.data) if self.bias is not None and self.bias.requires_grad else None

            for batch in range(n):
                for oc in range(out_channels):
                    for i in range(h_out):
                        for j in range(w_out):
                            h_start = i * sh
                            w_start = j * sw
                            grad_value = out.grad[batch, oc, i, j]

                            window = x_padded[
                                batch,
                                :,
                                h_start:h_start + kh,
                                w_start:w_start + kw
                            ]

                            if x.requires_grad:
                                dx_padded[
                                    batch,
                                    :,
                                    h_start:h_start + kh,
                                    w_start:w_start + kw
                                ] += self.weight.data[oc] * grad_value

                            if self.weight.requires_grad:
                                dw[oc] += window * grad_value

                            if db is not None:
                                db[0, oc, 0, 0] += grad_value

            if x.requires_grad:
                if ph == 0 and pw == 0:
                    x.grad += dx_padded
                else:
                    x.grad += dx_padded[:, :, ph:ph + h, pw:pw + w]

            if self.weight.requires_grad:
                self.weight.grad += dw

            if db is not None:
                self.bias.grad += db

        out._backward = _backward
        return out

    def __repr__(self):
        return (
            f"Conv2d(in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"bias={self.bias is not None})"
        )


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride if stride is not None else kernel_size)

    def forward(self, x):
        if x.data.ndim != 4:
            raise ValueError("MaxPool2d expects input shape (batch, channels, height, width)")

        n, c, h, w = x.data.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride

        h_out = (h - kh) // sh + 1
        w_out = (w - kw) // sw + 1

        out_data = np.zeros((n, c, h_out, w_out))

        for batch in range(n):
            for channel in range(c):
                for i in range(h_out):
                    for j in range(w_out):
                        h_start = i * sh
                        w_start = j * sw
                        window = x.data[
                            batch,
                            channel,
                            h_start:h_start + kh,
                            w_start:w_start + kw
                        ]
                        out_data[batch, channel, i, j] = np.max(window)

        out = Tensor(
            out_data,
            requires_grad=x.requires_grad,
            _children=(x,),
            _op="maxpool2d"
        )

        def _backward():
            if not x.requires_grad:
                return

            dx = np.zeros_like(x.data)

            for batch in range(n):
                for channel in range(c):
                    for i in range(h_out):
                        for j in range(w_out):
                            h_start = i * sh
                            w_start = j * sw
                            window = x.data[
                                batch,
                                channel,
                                h_start:h_start + kh,
                                w_start:w_start + kw
                            ]

                            max_value = np.max(window)
                            mask = window == max_value
                            count = mask.sum()

                            dx[
                                batch,
                                channel,
                                h_start:h_start + kh,
                                w_start:w_start + kw
                            ] += mask * out.grad[batch, channel, i, j] / count

            x.grad += dx

        out._backward = _backward
        return out

    def __repr__(self):
        return f"MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride})"


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride if stride is not None else kernel_size)

    def forward(self, x):
        if x.data.ndim != 4:
            raise ValueError("AvgPool2d expects input shape (batch, channels, height, width)")

        n, c, h, w = x.data.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride

        h_out = (h - kh) // sh + 1
        w_out = (w - kw) // sw + 1

        out_data = np.zeros((n, c, h_out, w_out))

        for batch in range(n):
            for channel in range(c):
                for i in range(h_out):
                    for j in range(w_out):
                        h_start = i * sh
                        w_start = j * sw
                        window = x.data[
                            batch,
                            channel,
                            h_start:h_start + kh,
                            w_start:w_start + kw
                        ]
                        out_data[batch, channel, i, j] = np.mean(window)

        out = Tensor(
            out_data,
            requires_grad=x.requires_grad,
            _children=(x,),
            _op="avgpool2d"
        )

        def _backward():
            if not x.requires_grad:
                return

            dx = np.zeros_like(x.data)
            scale = 1.0 / (kh * kw)

            for batch in range(n):
                for channel in range(c):
                    for i in range(h_out):
                        for j in range(w_out):
                            h_start = i * sh
                            w_start = j * sw

                            dx[
                                batch,
                                channel,
                                h_start:h_start + kh,
                                w_start:w_start + kw
                            ] += out.grad[batch, channel, i, j] * scale

            x.grad += dx

        out._backward = _backward
        return out

    def __repr__(self):
        return f"AvgPool2d(kernel_size={self.kernel_size}, stride={self.stride})"


class GlobalAvgPool2d(Module):
    def __init__(self, keepdims=True):
        super().__init__()
        self.keepdims = keepdims

    def forward(self, x):
        return x.mean(axis=(2, 3), keepdims=self.keepdims)

    def __repr__(self):
        return f"GlobalAvgPool2d(keepdims={self.keepdims})"
