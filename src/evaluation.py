
import numpy as np

# import sys
# from pathlib import Path

# # Add the project root to Python's import path.
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))
# from src.losses import softmax_cross_entropy
from .losses import softmax_cross_entropy

def evaluate(model, X, y, batch_size=8):
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for start in range(0, len(X), batch_size):
        X_batch = X[start:start + batch_size]
        y_batch = y[start:start + batch_size]

        logits = model.forward(X_batch, training=False)

        loss, _ = softmax_cross_entropy(logits, y_batch)

        count = len(X_batch)
        total_loss += loss * count
        total_correct += np.sum(
            np.argmax(logits, axis=1) == y_batch
        )
        total_seen += count

    if total_seen == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    return (
        total_loss / total_seen,
        total_correct / total_seen,
    )