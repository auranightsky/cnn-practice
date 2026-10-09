
import numpy as np


class Adam:
    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, parameters_and_grads):
        self.t += 1

        for parameter, gradient in parameters_and_grads:
            key = id(parameter)

            if key not in self.m:
                self.m[key] = np.zeros_like(parameter)
                self.v[key] = np.zeros_like(parameter)

            self.m[key] = (
                self.beta1 * self.m[key]
                + (1 - self.beta1) * gradient
            )

            self.v[key] = (
                self.beta2 * self.v[key]
                + (1 - self.beta2) * gradient**2
            )

            m_corrected = self.m[key] / (
                1 - self.beta1**self.t
            )

            v_corrected = self.v[key] / (
                1 - self.beta2**self.t
            )

            parameter -= self.lr * m_corrected / (
                np.sqrt(v_corrected) + self.epsilon
            )