from data_loader import load_cifar10_from_dir
from network import NeuralNetwork
from train import train, load_model_weights
from test import test
from hyperparameter_tuning import hyperparameter_tuning
from plot import plot_curves
import matplotlib.pyplot as plt


def plot_weights_distribution(model):
    for i, weight in enumerate(model.weights):
        plt.figure(figsize=(8, 4))
        plt.hist(weight.flatten(), bins=50, alpha=0.7, label=f'Layer {i+1} weights')
        plt.title(f'Weight Distribution of Layer {i+1}')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    x_train, y_train, x_val, y_val, x_test, y_test = load_cifar10_from_dir()
    best_hyper, training_losses, val_losses, val_acc = hyperparameter_tuning(x_train, y_train, x_val, y_val)
    print(best_hyper)
    lr = best_hyper['learning_rates']
    hidden = best_hyper['hidden_layers_sizes']
    reg = best_hyper['regularization_strengths']
    final_model = NeuralNetwork(input_size=3072, hidden_layers=hidden, output_size=10, reg_lambda=reg)
    training_losses, val_losses, val_accuracy = train(final_model, x_train, y_train, x_val, y_val, learning_rate=lr, epochs=20, batch_size=64)
    final_model.weights = load_model_weights()
    acc = test(final_model, x_test, y_test)
    print(f'test_acc:{acc}')
    plot_curves(training_losses, val_losses, val_accuracy, filename='final_png.png')
    plot_weights_distribution(final_model)
