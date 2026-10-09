
import numpy as np

def im2col(x, kernel_size, padding=0, stride=1):
    """
    x: (N, C, H, W)

    Returns:
        columns: (N * OH * OW, C * KH * KW)
        output height and width
    """
    N, C, H, W = x.shape
    KH, KW = kernel_size

    x_padded = np.pad(
        x,
        (
            (0, 0),
            (0, 0),
            (padding, padding),
            (padding, padding),
        ),
    )

    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1

    columns = []

    for n in range(N):
        for oy in range(OH):
            for ox in range(OW):
                patch = x_padded[
                    n,
                    :,
                    oy * stride:oy * stride + KH,
                    ox * stride:ox * stride + KW,
                ]
                columns.append(patch.reshape(-1))

    return np.asarray(columns), OH, OW


def col2im(columns, x_shape, kernel_size, padding=0, stride=1):
    """Reverse im2col by accumulating overlapping image patches."""
    N, C, H, W = x_shape
    KH, KW = kernel_size

    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1

    dx_padded = np.zeros(
        (N, C, H + 2 * padding, W + 2 * padding),
        dtype=columns.dtype,
    )

    columns = columns.reshape(N, OH, OW, C, KH, KW)

    for n in range(N):
        for oy in range(OH):
            for ox in range(OW):
                y = oy * stride
                x = ox * stride

                dx_padded[
                    n, :, y:y + KH, x:x + KW
                ] += columns[n, oy, ox]

    if padding > 0:
        return dx_padded[
            :, :, padding:-padding, padding:-padding
        ]

    return dx_padded


class Conv2D:
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        seed=None,
    ):
        self.kernel_size = (
            (kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else tuple(kernel_size)
        )

        self.stride = stride
        self.padding = padding

        rng = np.random.default_rng(seed)
        kh, kw = self.kernel_size

        # He initialization for ReLU-based networks.
        scale = np.sqrt(2.0 / (in_channels * kh * kw))

        self.W = (
            rng.standard_normal(
                (out_channels, in_channels, kh, kw)
            ) * scale
        ).astype(np.float32)

        self.b = np.zeros(out_channels, dtype=np.float32)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x_shape = x.shape

        self.columns, self.OH, self.OW = im2col(
            x,
            self.kernel_size,
            self.padding,
            self.stride,
        )

        weight_matrix = self.W.reshape(self.W.shape[0], -1)

        output = self.columns @ weight_matrix.T
        output += self.b

        N = x.shape[0]
        F = self.W.shape[0]

        return output.reshape(
            N, self.OH, self.OW, F
        ).transpose(0, 3, 1, 2)

    def backward(self, dout):
        # dout: (N, out_channels, OH, OW)
        N = dout.shape[0]

        dout_rows = dout.transpose(
            0, 2, 3, 1
        ).reshape(-1, self.W.shape[0])

        self.dW = (
            dout_rows.T @ self.columns
        ).reshape(self.W.shape)

        self.db = dout_rows.sum(axis=0)

        weight_matrix = self.W.reshape(
            self.W.shape[0], -1
        )

        dcolumns = dout_rows @ weight_matrix

        dx = col2im(
            dcolumns,
            self.x_shape,
            self.kernel_size,
            self.padding,
            self.stride,
        )

        return dx

    def parameters_and_grads(self):
        return [
            (self.W, self.dW),
            (self.b, self.db),
        ]