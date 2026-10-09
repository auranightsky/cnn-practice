
import numpy as np


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return np.maximum(0, x)

    def backward(self, dout):
        return dout * self.mask

    def parameters_and_grads(self):
        return []


class Flatten:
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.input_shape)

    def parameters_and_grads(self):
        return []


class Dense:
    def __init__(self, in_features, out_features, seed=None):
        rng = np.random.default_rng(seed)

        scale = np.sqrt(2.0 / in_features)

        self.W = (
            rng.standard_normal(
                (in_features, out_features)
            ) * scale
        ).astype(np.float32)

        self.b = np.zeros(out_features, dtype=np.float32)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = dout.sum(axis=0)

        dx = dout @ self.W.T
        return dx

    def parameters_and_grads(self):
        return [
            (self.W, self.dW),
            (self.b, self.db),
        ]


class Dropout:
    def __init__(self, p=0.3, seed=None):
        if not 0 <= p < 1:
            raise ValueError("p must be in [0, 1).")

        self.p = p
        self.rng = np.random.default_rng(seed)

    def forward(self, x, training=True):
        if not training or self.p == 0:
            self.mask = None
            return x

        # Inverted dropout: surviving activations are
        # scaled during training, not during inference.
        keep_probability = 1.0 - self.p

        self.mask = (
            self.rng.random(x.shape) < keep_probability
        ).astype(x.dtype) / keep_probability

        return x * self.mask

    def backward(self, dout):
        if self.mask is None:
            return dout

        return dout * self.mask

    def parameters_and_grads(self):
        return []