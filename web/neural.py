import numpy as np
from flask import Blueprint, jsonify, request

import image_processing.image as image
from database.Database import Database
from neural_service.neuralNetwork import NeuralNetwork

neural = Blueprint('neural', __name__)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_der(z):
    return sigmoid(z) * (1 - sigmoid(z))


lr = 0.15
nn = NeuralNetwork([784, 30, 10], sigmoid, sigmoid_der, lr)
db = Database('mongodb://localhost:27017/')


@neural.route('/predict', methods=['POST'])
def show():
    array = np.fromstring(request.files['image'].read(), np.uint8)
    features = image.load_image_from_array(array)
    weights, biases = db.load_weights_biases()
    nn.b = biases
    nn.w = weights
    out = nn.forward_propagation(features)
    number = np.argmax(out)
    probability = np.max(out)
    return jsonify(
        {"number": int(number),
         'probability': probability * 100}
    )
