import numpy as np
from sklearn.utils import shuffle
from data_preprocessing import apply_noise, get_model_samples, mask_variable, get_adj_matrix
from analysis import create_dataframe
from seed_finder import SeedFinder
from center_finder import CenterFinder


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
        return model_variables, (X, y, en)

    def data_for_seed_finder(self, data_type="train"):
        """
        Loads the data and transforms it for the seed finder network.
        """
        model_variables, _ = self.load_and_prepare_data(data_type=data_type)
        X, _, is_seed, _, _ = model_variables

        # reshape variables for the model
        X = X.reshape(-1, 7, 7, 1)
        is_seed = is_seed.reshape(-1)

        # remove non-existant windows (i.e. windows added during padding)
        X = X[is_seed != -1]
        is_seed = is_seed[is_seed != -1]

        return X, is_seed
    
    def data_for_center_finder(self, data_type="train", threshold=0.3, n=4, **kwargs):
        """
        Loads the data and transforms it for the center finder network.
        Args:
            - threshold: float, threshold for the seed finder network
            - data_type: str, 'train', 'valid', or 'test'
            - n: int, number of input windows to the network
            - **kwargs: keyword arguments for the seed finder network paths
        """
        model_variables, _ = self.load_and_prepare_data(data_type=data_type)
        X, indices, is_seed, y, en = model_variables # samples are combined by event with padding 35

        # make seed finder predictions
        ypr = SeedFinder().prediction(X, **kwargs)

        var = [X, indices, is_seed, y, en, ypr]
        for i, _ in enumerate(var):
            var[i] = mask_variable(var[i], ypr, threshold=threshold, n=n)    

        # shuffle events as now they are ordered by the probability predicted by the first network
        for i, _ in enumerate(X):
            var[0][i], var[1][i], var[2][i], var[3][i], var[4][i], var[5][i] = shuffle(var[0][i], var[1][i], 
                                                                                       var[2][i], var[3][i], 
                                                                                       var[4][i], var[5][i])
        # create a mask for the true coordinates and energy
        mask = var[3][:,:, 0] > 0
        # change the coordinate position, so it is relative to the center of the 7x7 image
        var[3][mask] = var[3][mask] - var[1][mask].astype(float) + 3. - 3.5
        # normalize the energy
        var[4][mask] = var[4][mask]/np.max(var[4][mask])

        # delete the events where no clusters were chosen or no clusters exist after selection
        condition = ((np.sum(var[3][:,:,0], axis=1) != 0.) & (np.sum(var[2] != -1, axis=1) != 0))
        for i, _ in enumerate(var):
            var[i] = var[i][condition]

        # get adjacency matrix
        adj_matrix, adj_coef = get_adj_matrix(var[0], var[1], n=n)

        # reshape variables for the model
        var[0] = np.swapaxes(var[0], 1, 3)
        var[4] = np.expand_dims(var[4], axis=2)
        var[2] = np.expand_dims(var[2], axis=2)

        return (var[0], adj_matrix, adj_coef), {'center':var[3], 'energy': var[4], 'seed': var[2]} 
    
    def prediction(self, data_type="test", threshold=0.3, n=4, **kwargs):
        """
        Performs the full analysis of the data, from loading to the final predictions.
        Args:
            - threshold: float, threshold for the seed finder network
            - data_type: str, 'train', 'valid', or 'test'
            - n: int, number of input windows to the network
            - **kwargs: keyword arguments for the seed and center finder network paths
        """
        # load the data
        model_variables, initial_variables = self.load_and_prepare_data(data_type=data_type)
        X, indices, is_seed, y, en = model_variables # samples are combined by event with padding 35

        # make seed finder predictions
        seed_pr = SeedFinder().prediction(X, **kwargs)

        var = [X, indices, is_seed, y, en, seed_pr]
        for i, _ in enumerate(var):
            var[i] = mask_variable(var[i], seed_pr, threshold=threshold, n=n)

        # delete the events where no clusters were chosen or no clusters exist after selection
        condition = ((np.sum(var[3][:,:,0], axis=1) != 0.) & (np.sum(var[2] != -1, axis=1) != 0))
        for i, _ in enumerate(var):
            var[i] = var[i][condition]

        # get adjacency matrix
        adj_matrix, adj_coef = get_adj_matrix(var[0], var[1], n=n)

        # reshape variables for the model
        var[0] = np.swapaxes(var[0], 1, 3) 

        # get the predictions from the center finder network
        center_pr = CenterFinder({}).prediction(var[0], adj_matrix, adj_coef, **kwargs)

        yc, en, ys = center_pr['center'], center_pr['energy'], center_pr['seed']

        # change back the coordinate position and energy
        yc = yc + 3.5 + var[1] - 7//2
        en *= 100
        return (yc, en, ys, seed_pr), initial_variables
    
    def analysis(self, data_type="test", threshold=0.3, n=4, out_path='data', **kwargs):
        """
        Performs the full analysis of the data, from loading to combining the final predictions in the dataframe.
        Args:
            - threshold: float, threshold for the seed finder network
            - data_type: str, 'train', 'valid', or 'test'
            - n: int, number of input windows to the network
            - **kwargs: keyword arguments for the seed and center finder network paths
        """

        # get the model predictions
        pred, true = self.prediction(data_type=data_type, threshold=threshold, n=n, **kwargs)
        df_true = create_dataframe(pred, true)
        return df_true
    
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
