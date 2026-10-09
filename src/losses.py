
import numpy as np


def softmax(logits):
    """Convert scores into probabilities."""
    shifted = logits - np.max(
        logits, axis=1, keepdims=True
    )

    exp_values = np.exp(shifted)

    return exp_values / np.sum(
        exp_values, axis=1, keepdims=True
    )


def softmax_cross_entropy(logits, labels):
    """
    Args:
        logits: (N, num_classes), raw model scores
        labels: (N,), integer class labels

    Returns:
        loss: scalar mean cross-entropy
        dlogits: gradient of the mean loss
    """
    N = logits.shape[0]

    shifted = logits - np.max(
        logits, axis=1, keepdims=True
    )

    log_sum_exp = np.log(
        np.sum(np.exp(shifted), axis=1)
    )

    log_probabilities = (
        shifted - log_sum_exp[:, None]
    )

    loss = -np.mean(
        log_probabilities[np.arange(N), labels]
    )

    probabilities = np.exp(log_probabilities)
    dlogits = probabilities.copy()

    dlogits[np.arange(N), labels] -= 1.0
    dlogits /= N

    return float(loss), dlogits


def accuracy(logits, labels):
    predictions = np.argmax(logits, axis=1)
    return float(np.mean(predictions == labels))