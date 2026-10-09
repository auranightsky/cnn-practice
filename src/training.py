
import numpy as np
from .losses import softmax_cross_entropy
from .evaluation import evaluate

def train(
    model,
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    epochs=1,
    batch_size=8,
    learning_rate=0.001,
    seed=42,
):
    rng = np.random.default_rng(seed)

    from .optimizer import Adam
    optimizer = Adam(learning_rate=learning_rate)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    for epoch in range(epochs):
        indices = rng.permutation(len(X_train))

        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for start in range(0, len(X_train), batch_size):
            batch_indices = indices[start:start + batch_size]

            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Forward pass
            logits = model.forward(X_batch, training=True)

            loss, dlogits = softmax_cross_entropy(
                logits, y_batch
            )

            # Backward pass
            model.backward(dlogits)

            # Update all trainable parameters
            optimizer.step(model.parameters_and_grads())

            # Record metrics
            batch_count = len(X_batch)
            total_loss += loss * batch_count
            total_correct += np.sum(
                np.argmax(logits, axis=1) == y_batch
            )
            total_seen += batch_count

        train_loss = total_loss / total_seen
        train_accuracy = total_correct / total_seen

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        message = (
            f"Epoch {epoch + 1}/{epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"train accuracy: {train_accuracy:.2%}"
        )

        if X_test is not None and y_test is not None:
            test_loss, test_accuracy = evaluate(
                model, X_test, y_test,
                batch_size=batch_size,
            )

            history["test_loss"].append(test_loss)
            history["test_accuracy"].append(test_accuracy)

            message += (
                f" | test loss: {test_loss:.4f}"
                f" | test accuracy: {test_accuracy:.2%}"
            )

        print(message)

    return history