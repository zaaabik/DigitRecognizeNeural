import datetime as dt
import optparse as op

import numpy as np

from data_loader import mnist_loader as loader
from database.Database import Database
from image_processing import image
from neural_service import neuralNetwork

cmd_parser = op.OptionParser()
cmd_parser.add_option('--mode', '-m')
options, args = cmd_parser.parse_args()
lr = 0.2
epochs = 30
mini_batch_size = 10


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_der(z):
    return sigmoid(z) * (1 - sigmoid(z))


nn = neuralNetwork.NeuralNetwork([784, 30, 10], sigmoid, sigmoid_der, lr)
db = Database('mongodb://localhost:27017/')


def mse(x, y):
    return np.sum((x - y) ** 2)


def learn():
    pass


def real_learn():
    start = dt.datetime.now()
    x_train, y_train, x_test, y_test = loader.load_data_wrapper()
    training_data = list(zip(x_train, y_train))
    n = len(training_data)
    predicted = nn.forward_propagation(x_train)
    print(mse(predicted, y_train))
    test_predicted_numbers = nn.forward_propagation(x_test).argmax(axis=1)
    errors_count = (y_test == test_predicted_numbers).sum()
    print("correct before = ", errors_count)

    for i in range(0, epochs):
        np.random.shuffle(training_data)

        for j in range(0, n, mini_batch_size):
            nn.backward_propagation(x_train[j:j + mini_batch_size], y_train[j:j + mini_batch_size])
        predicted = nn.forward_propagation(x_train)
        print(f"mse {mse(predicted, y_train)}")
        test_results = [(np.argmax(nn.forward_propagation(x)), y)
                        for (x, y) in zip(x_test, y_test)]
        correct = sum(int(x == y) for (x, y) in test_results)
        print("correct Epoch {} : {}".format(i, correct))

    test_predicted_numbers = nn.forward_propagation(x_test).argmax(axis=1)
    errors_count = (y_test == test_predicted_numbers).sum()
    print("correct after = ", errors_count)

    db.save_weights_biases(nn.w, nn.b)
    end = dt.datetime.now()
    print("time ", (end - start).seconds)


def predict():
    weights, biases = db.load_weights_biases()
    nn.w = weights
    nn.b = biases
    _, _, x_test, y_test = loader.load_data_wrapper()
    n = x_test.shape[0]
    r = np.random.randint(0, n)
    result = y_test[r]
    features = x_test[r]
    image.create_image(features, (28, 28), 'created.bmp')
    out = nn.forward_propagation(features)
    nn_res = np.argmax(out)
    # print(out)
    print("nn out ", nn_res)
    print("out ", result)


def predict_image_by_path():
    weights, biases = db.load_weights_biases()
    nn.w = weights
    nn.b = biases
    features = image.load_image('test.png')
    out = nn.forward_propagation(features)
    number = np.argmax(out)
    print(out)
    print(number)


def predict_image(array):
    features = image.get_features_from_array(array)
    out = nn.forward_propagation(features)
    number = np.argmax(out)
    return number


def main():
    if int(options.mode) == 1:
        learn()
    elif int(options.mode) == 2:
        real_learn()
    elif int(options.mode) == 3:
        predict()
    elif int(options.mode) == 4:
        predict_image_by_path()
    else:
        raise Exception('wrong mode')


if __name__ == '__main__':
    main()
