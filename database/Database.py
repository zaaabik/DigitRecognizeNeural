import pickle

import numpy as np
from bson.binary import Binary
from pymongo import MongoClient


class Database:
    WEIGHTS_NAME = 'weights'
    BIASES_NAME = 'biases'
    DATABASE_NAME = 'neural'
    COLLECTION_NAME = 'digitRecognize'
    ID = 'params'

    def __init__(self, connection_string):
        self.__client = MongoClient(connection_string)

    def save_weights_biases(self, weights, biases):
        db = self.__client[self.DATABASE_NAME]
        collection = db[self.COLLECTION_NAME]
        collection.update(
            {'_id': self.ID},
            {self.WEIGHTS_NAME: Binary(pickle.dumps(np.array(weights), protocol=2)),
             self.BIASES_NAME: Binary(pickle.dumps(np.array(biases), protocol=2))},
            True
        )

    def load_weights_biases(self):
        db = self.__client[self.DATABASE_NAME]
        collection = db[self.COLLECTION_NAME]
        params = collection.find_one(
            {'_id': self.ID}
        )
        biases = params[self.BIASES_NAME]
        weights = params[self.WEIGHTS_NAME]
        biases = pickle.loads(biases)
        weights = pickle.loads(weights)
        return weights, biases
