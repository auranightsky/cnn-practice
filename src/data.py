from pathlib import Path
from urllib.request import urlretrieve
import gzip
import struct
import numpy as np


DATA_DIR = Path("data/mnist")

BASE_URL = (
    "https://storage.googleapis.com/"
    "cvdf-datasets/mnist/"
)


def download_file(filename):
    """Download one compressed MNIST file if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    path = DATA_DIR / filename

    if not path.exists():
        url = BASE_URL + filename
        print(f"Downloading {filename}...")
        urlretrieve(url, path)

    return path


def read_images(filename):
    """Read MNIST images from a gzipped IDX file."""
    path = download_file(filename)

    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(
            ">IIII", f.read(16)
        )

        if magic != 2051:
            raise ValueError("Invalid MNIST image file.")

        data = np.frombuffer(
            f.read(), dtype=np.uint8
        ).copy()

    images = data.reshape(count, 1, rows, cols)
    return images.astype(np.float32) / 255.0


def read_labels(filename):
    """Read MNIST labels from a gzipped IDX file."""
    path = download_file(filename)

    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))

        if magic != 2049:
            raise ValueError("Invalid MNIST label file.")

        labels = np.frombuffer(
            f.read(), dtype=np.uint8
        ).copy()

    return labels.astype(np.int64)


def load_mnist(
    train_size=1000,
    test_size=200,
    seed=42,
):
    """
    Return:
        X_train: (train_size, 1, 28, 28)
        y_train: (train_size,)
        X_test:  (test_size, 1, 28, 28)
        y_test:  (test_size,)
    """
    X_train = read_images("train-images-idx3-ubyte.gz")
    y_train = read_labels("train-labels-idx1-ubyte.gz")

    X_test = read_images("t10k-images-idx3-ubyte.gz")
    y_test = read_labels("t10k-labels-idx1-ubyte.gz")

    rng = np.random.default_rng(seed)

    train_indices = rng.permutation(len(X_train))[:train_size]
    test_indices = rng.permutation(len(X_test))[:test_size]

    return (
        X_train[train_indices],
        y_train[train_indices],
        X_test[test_indices],
        y_test[test_indices],
    )


def one_hot(labels, num_classes=10):
    """Convert integer labels into one-hot vectors."""
    result = np.zeros(
        (len(labels), num_classes),
        dtype=np.float32,
    )

    result[np.arange(len(labels)), labels] = 1.0
    return result