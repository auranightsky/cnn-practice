
import numpy as np
from .convolution import Conv2D
from .pooling import MaxPool2D
from .layers import ReLU, Flatten, Dense, Dropout

class NumpyCNN:
    def __init__(self, seed=42):
        self.conv1 = Conv2D(
            1, 4, kernel_size=3, padding=1, seed=seed
        )
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(2)

        self.conv2 = Conv2D(
            4, 8, kernel_size=3, padding=1, seed=seed + 1
        )
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(2)

        self.flatten = Flatten()
        self.fc1 = Dense(8 * 7 * 7, 32, seed=seed + 2)
        self.relu3 = ReLU()
        self.dropout = Dropout(p=0.3, seed=seed + 3)
        self.fc2 = Dense(32, 10, seed=seed + 4)

    def forward(self, x, training=False):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        x = self.flatten.forward(x)
        x = self.fc1.forward(x)
        x = self.relu3.forward(x)
        x = self.dropout.forward(x, training=training)
        x = self.fc2.forward(x)

        # Return logits, not softmax probabilities.
        return x

    def backward(self, dout):
        dout = self.fc2.backward(dout)
        dout = self.dropout.backward(dout)
        dout = self.relu3.backward(dout)
        dout = self.fc1.backward(dout)
        dout = self.flatten.backward(dout)

        dout = self.pool2.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.conv2.backward(dout)

        dout = self.pool1.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.conv1.backward(dout)

        return dout

    def parameters_and_grads(self):
        layers = [
            self.conv1,
            self.conv2,
            self.fc1,
            self.fc2,
        ]

        result = []

        for layer in layers:
            result.extend(layer.parameters_and_grads())

        return result

    def predict(self, x):
        logits = self.forward(x, training=False)
        return np.argmax(logits, axis=1)