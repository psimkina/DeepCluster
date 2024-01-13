from tensorflow import keras
from keras import layers


class SeedFinder:
    """
    SeedFinder class for the seed finder network.
    """

    def __init__(self, crop_size=7):
        pass

    def architecture(self, crop_size=7):
        """
        Function to define the architecture of the seed finder network.
        Args:
            - crop_size: int, size of the crop window
        """

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
    
    def prediction(self, X_crop, **kwargs): 
        """
        Returns the prediction of the seed finder network.
        Args:
            - X_crop: np.array, cropped input image
            - model_path: str, path to the model
            - weight_path: str, path to the weights
        """
        model_path = kwargs.get("model_path", None)
        weight_path = kwargs.get("weight_path", None)
        
        if model_path is None:
            model = self.architecture()
        else:
            model = keras.models.load_model(model_path)
        if weight_path is not None:
            model.load_weights(weight_path)
        
        # reshape X_crop for the model
        n = X_crop.shape[1]
        X_crop = X_crop.reshape(-1, 7, 7, 1)
        ypr = model.predict(X_crop)

        if n != 7: # for the case when samples are combined by event, e.g. with padding 35
            ypr = ypr.reshape(-1, n, 1)
        return ypr