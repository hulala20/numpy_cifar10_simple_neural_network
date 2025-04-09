import numpy as np
import pickle
from test import test


def save_model_weights(weights, filename='best_model_weights.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(weights, f)
    print(f'Model weights saved to {filename}')


def load_model_weights(filename='best_model_weights.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def train(model, X_train, y_train, X_val, y_val, learning_rate=0.01, epochs=100, batch_size=32):
    n = X_train.shape[0]
    best_weights = None
    best_loss = float('inf')
    training_losses = []
    val_losses = []
    val_accuracy = []

    for epoch in range(epochs):
        # 打乱数据
        indices = np.arange(n)
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        for i in range(0, n, batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            y_hat = model.forward(X_batch)
            loss = model.compute_loss(y_hat, y_batch)
            model.backward(X_batch, y_batch)
            model.update_parameters(learning_rate)

        val_y_hat = model.forward(X_val)
        val_loss = model.compute_loss(val_y_hat, y_val)
        val_acc = test(model, X_val, y_val)
        val_losses.append(val_loss)
        training_losses.append(loss)
        val_accuracy.append(val_acc)

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = [w.copy() for w in model.weights]
            save_model_weights(best_weights)

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}')

    model.weights = best_weights
    return training_losses, val_losses, val_accuracy
