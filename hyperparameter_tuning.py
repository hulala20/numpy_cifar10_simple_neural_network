from network import NeuralNetwork
from train import train
from test import test
from plot import plot_curves


def hyperparameter_tuning(X_train, y_train, X_val, y_val):
    learning_rates = [0.01,0.001]
    hidden_layers_sizes = [[128, 256]]
    regularization_strengths = [0.0, 0.01]
    best_hyper = {}
    min_val_loss = 1e10

    for lr in learning_rates:
        for hidden in hidden_layers_sizes:
            for reg in regularization_strengths:
                print(f'Training with lr: {lr}, Hidden Layers: {hidden}, Reg: {reg}')
                model = NeuralNetwork(input_size=3072, hidden_layers=hidden,
                                      output_size=10, reg_lambda=reg)
                training_losses, val_losses, val_acc = train(model, X_train,
                                                             y_train, X_val, y_val, learning_rate=lr, epochs=20, batch_size=64)
                if min(val_losses) < min_val_loss:
                    min_val_loss = min(val_losses)
                    best_hyper['learning_rates'] = lr
                    best_hyper['hidden_layers_sizes'] = hidden
                    best_hyper['regularization_strengths'] = reg
                acc = test(model, X_val, y_val)
                plot_curves(train_losses=training_losses, val_losses=val_losses, val_accuracies=val_acc)
    return best_hyper, training_losses, val_losses, val_acc
