import numpy as np
from data_preprocessing import apply_noise, get_model_samples


class Particle:
    """
    Class for particle data.

    Args:
        - self.particle_type: int, 0 for photon, 1 for electron, 2 for pion
        - self.n_pcl: int, number of particles in the event
        - self.path: str, path to the data folder
    """

    def __init__(self, particle_type) -> None:
        self.particle_type = particle_type

    def data_path(self):
        """
        Returns the path to the data folder.
        """
        particle_dict = {0: "photon", 1: "electron", 2: "pion"}

        path = "../data/" + particle_dict[self.particle_type] + "/"
        return path

    def load_data(self, data_type="train"):
        """
        Loads the data from the data folder.
        Args:
            - type: str, 'train', 'valid', or 'test'
        Returns:
            - X: np.array, shape=(n_events, 51, 51)
            - y: np.array, shape=(n_events, n_pcl, 2)
            - en: np.array, shape=(n_events, n_pcl, 1)
        """
        self.path = self.data_path()

        X = np.load(self.path + "X{}.npy".format(data_type))
        y = np.load(self.path + "y{}.npy".format(data_type))
        en = np.load(self.path + "en{}.npy".format(data_type))

        # change the shape of y and en for consistency
        if len(y.shape) != 3:
            y = np.expand_dims(y, axis=1)
            en = np.expand_dims(en, axis=1)
        return X, y, en

    def load_and_prepare_data(self, data_type="train"):
        """
        Loads the data and applies noise.
        Args:
            - type: str, 'train', 'valid', or 'test'
        Returns:
            - X: np.array, shape=(n_events, 51, 51)
            - y: np.array, shape=(n_events, n_pcl, 2)
            - en: np.array, shape=(n_events, n_pcl, 1)
        """
        X, y, en = self.load_data(data_type=data_type)
        X = apply_noise(X)
        model_variables = get_model_samples(X, y, en)
        return model_variables

    def data_for_seed_finder(self, data_type="train"):
        """
        Loads the data and transforms it for the seed finder network.
        """
        model_variables = self.load_and_prepare_data(data_type=data_type)
        X, _, is_seed, _, _ = model_variables

        # reshape variables for the model
        X = X.reshape(-1, 7, 7, 1)
        is_seed = is_seed.reshape(-1)

        # remove non-existant windows (i.e. windows added during padding)
        X = X[is_seed != -1]
        is_seed = is_seed[is_seed != -1]
        print(is_seed.shape, X.shape)
        return X, is_seed
    
    def data_for_center_finder(self, data_type="train"):
        """
        Loads the data and transforms it for the center finder network.
        """
        model_variables = self.load_and_prepare_data(data_type=data_type)
        #return X, (y, en, is_seed)


class Photon(Particle):
    """
    Class for photon data.

    Args:
        - self.n_pcl: int, number of particles in the event
        - self.path: str, path to the data folder
    """

    def __init__(self, n_pcl=1) -> None:
        super().__init__(particle_type=0)
        self.n_pcl = n_pcl

    def data_path(self):
        """
        Returns the path to the data folder.
        """
        if self.n_pcl == 1:
            path = "../data/photon/" + "one_particle/"
        else:
            path = "../data/photon/" + "two_particles/"
        return path
