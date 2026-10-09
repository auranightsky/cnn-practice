
import unittest
import numpy as np
from src.pooling import MaxPool2D

class TestMaxPool2D(unittest.TestCase):
    def test_forward_values(self):
        x = np.array(
            [[[
                [1, 3, 2, 4],
                [5, 6, 1, 2],
                [7, 8, 9, 0],
                [1, 2, 3, 4],
            ]]],
            dtype=np.float32,
        )

        layer = MaxPool2D(kernel_size=2)
        output = layer.forward(x)

        expected = np.array(
            [[[
                [6, 4],
                [8, 9],
            ]]],
            dtype=np.float32,
        )

        np.testing.assert_array_equal(output, expected)

    def test_backward_shape(self):
        x = np.random.randn(2, 3, 8, 8).astype(np.float32)

        layer = MaxPool2D(2)
        output = layer.forward(x)

        dx = layer.backward(np.ones_like(output))

        self.assertEqual(dx.shape, x.shape)

    def test_gradient_reaches_maximum(self):
        x = np.array(
            [[[[1, 2], [3, 9]]]],
            dtype=np.float32,
        )

        layer = MaxPool2D(2)
        output = layer.forward(x)
        dx = layer.backward(np.ones_like(output))

        expected = np.array(
            [[[[0, 0], [0, 1]]]],
            dtype=np.float32,
        )

        np.testing.assert_array_equal(dx, expected)


if __name__ == "__main__":
    unittest.main()