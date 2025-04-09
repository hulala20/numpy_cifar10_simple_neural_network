import numpy as np


def test(model, X_test, y_test):
    y_hat = model.forward(X_test)
    predictions = np.argmax(y_hat, axis=1)
    ground_truth = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == ground_truth)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    return accuracy
