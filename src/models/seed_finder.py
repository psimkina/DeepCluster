import numpy as np
from tqdm import tqdm

from tensorflow import keras
from keras import layers


class SeedFinder:
    '''
    SeedFinder class for the seed finder network.
    '''
    def __init__(self, crop_size=7):
        pass

    def architecture(self, crop_size=7):
        '''
        Function to define the architecture of the seed finder network.
        Args:
            - crop_size: int, size of the crop window
        '''

        inputs = keras.Input(shape=(crop_size, crop_size, 1))
        x = layers.Conv2D(128, 3, activation=layers.LeakyReLU())(inputs)
        # x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, 3, activation=layers.LeakyReLU())(x)
        # x = layers.BatchNormalization()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(2400, activation=layers.LeakyReLU())(x)
        x = layers.Dense(500, activation=layers.LeakyReLU())(x)
        x = layers.Dense(1, activation="sigmoid")(x)

        model = keras.Model(inputs=inputs, outputs=x)

        return model