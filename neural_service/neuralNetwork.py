import numpy as np
from sklearn.model_selection import train_test_split


class NeuralNetwork:
    def __init__(self, neurons_on_layer_count, act_func, act_func_der, lr):
        self.w = []
        self.b = []
        self.act = act_func
        self.act_der = act_func_der
        self.lr = lr
        for i in range(1, len(neurons_on_layer_count)):
            self.w.append(np.random.randn(neurons_on_layer_count[i], neurons_on_layer_count[i - 1]))
            self.b.append(np.random.randn(neurons_on_layer_count[i], 1))

    def forward_propagation(self, input) -> np.ndarray:
        for b, w in zip(self.b, self.w):
            input = self.act(input @ w.T + b.T)
        return input

    def backward_propagation(self, input, result):
        m = input.shape[0]
        z = []
        activation = input
        a = [input]
        for b, w in zip(self.b, self.w):
            layer = activation @ w.T + b.T
            z.append(layer)
            activation = self.act(layer)
            a.append(activation)

        cur_error = (a[-1] - result)
        error = [cur_error]
        for i in reversed(range(1, len(self.w))):
            cur_error = cur_error @ self.w[i] * self.act_der(z[i - 1])
            error.append(cur_error)
        error.reverse()

        delta_w = []
        delta_b = []
        for i in range(0, len(error)):
            err = error[i]
            act = a[i]
            delta = err.T @ act
            delta_w.append(delta)
            delta_b.append(np.sum(err))
        delta_w = np.array(delta_w)
        delta_b = np.array(delta_b)
        for i in range(0, len(delta_w)):
            self.w[i] -= delta_w[i] * self.lr
            self.b[i] -= delta_b[i] * self.lr

    @staticmethod
    def split_data(data, train_percent):
        return np.array(train_test_split(data, train_size=train_percent))

    def prepare_dataset(self, file, output):
        data_set = np.genfromtxt(file, delimiter=',')
        data_set = data_set / np.max(data_set, axis=0)
        np.savetxt(output, data_set, delimiter=',', fmt='%f')

    def read_data_set(self, file):
        data_set = np.genfromtxt(file, delimiter=',')
        np.random.shuffle(data_set)
        return data_set[..., :-1], data_set[..., -1:]
