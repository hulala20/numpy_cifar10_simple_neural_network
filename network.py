import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, activation='relu', reg_lambda=0.0):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation = activation
        self.reg_lambda = reg_lambda

        # 权重初始化
        self.weights = []
        self.biases = []
        layer_sizes = [input_size] + hidden_layers + [output_size]

        for i in range(len(layer_sizes) - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01
            bias = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def activation_function(self, Z, derivative=False):
        if self.activation == 'relu':
            if derivative:
                return (Z > 0).astype(float)
            return np.maximum(0, Z)
        elif self.activation == 'sigmoid':
            if derivative:
                A = 1 / (1 + np.exp(-Z))
                return A * (1 - A)
            return 1 / (1 + np.exp(-Z))

    def forward(self, X):
        self.Z = []
        self.A = [X]

        for i in range(len(self.weights)):
            Z = np.dot(self.A[i], self.weights[i]) + self.biases[i]
            self.Z.append(Z)
            A = self.activation_function(Z)
            self.A.append(A)

        return self.A[-1]  # 返回输出层的激活值

    def backward(self, X, y):
        m = y.shape[0]
        self.d_weights = [0] * len(self.weights)
        self.d_biases = [0] * len(self.biases)

        # 计算输出层的误差
        dA = self.A[-1] - y
        for i in reversed(range(len(self.weights))):
            dZ = dA * self.activation_function(self.Z[i], derivative=True)
            self.d_weights[i] = np.dot(self.A[i].T, dZ) / m + (self.reg_lambda * self.weights[i] / m)
            self.d_biases[i] = np.sum(dZ, axis=0, keepdims=True) / m

            dA = np.dot(dZ, self.weights[i].T)

    def update_parameters(self, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.d_weights[i]
            self.biases[i] -= learning_rate * self.d_biases[i]

    def compute_loss(self, y_hat, y):
        m = y.shape[0]
        cross_entropy_loss = -np.sum(y * np.log(y_hat + 1e-12)) / m
        l2_loss = sum(np.sum(w**2) for w in self.weights) * (self.reg_lambda / (2 * m))
        return cross_entropy_loss + l2_loss
