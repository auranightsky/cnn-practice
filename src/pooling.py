import numpy as np
class MaxPool2D:
    def __init__(self, kernel_size=2, stride=None):
        self.kernel_size = kernel_size
        self.stride = (
            kernel_size if stride is None else stride
        )

    def forward(self, x):
        # x: (N, C, H, W)
        self.x_shape = x.shape

        N, C, H, W = x.shape
        K = self.kernel_size
        S = self.stride

        OH = (H - K) // S + 1
        OW = (W - K) // S + 1

        output = np.zeros(
            (N, C, OH, OW), dtype=x.dtype
        )
        self.argmax = np.zeros(
            (N, C, OH, OW), dtype=np.int64
        )

        for oy in range(OH):
            for ox in range(OW):
                patch = x[
                    :,
                    :,
                    oy * S:oy * S + K,
                    ox * S:ox * S + K,
                ]

                flattened = patch.reshape(N, C, K * K)

                self.argmax[:, :, oy, ox] = np.argmax(
                    flattened, axis=2
                )

                output[:, :, oy, ox] = np.max(
                    flattened, axis=2
                )

        return output

    def backward(self, dout):
        N, C, H, W = self.x_shape
        K = self.kernel_size
        S = self.stride

        OH, OW = dout.shape[2:]

        dx = np.zeros(
            (N, C, H, W), dtype=dout.dtype
        )

        for oy in range(OH):
            for ox in range(OW):
                indices = self.argmax[:, :, oy, ox]

                for ky in range(K):
                    for kx in range(K):
                        index = ky * K + kx
                        mask = indices == index

                        dx[
                            :,
                            :,
                            oy * S + ky,
                            ox * S + kx,
                        ] += dout[:, :, oy, ox] * mask

        return dx

    def parameters_and_grads(self):
        return []