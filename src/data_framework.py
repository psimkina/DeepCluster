
import numpy as np

class Particle:
    '''
    Class for particle data.
    
    Args: 
        - self.particle_type: int, 0 for photon, 1 for electron, 2 for pion
        - self.n_pcl: int, number of particles in the event
        - self.path: str, path to the data folder
    '''

    def __init__(self, particle_type) -> None:
        self.particle_type = particle_type
        
    def data_path(self):
        '''
        Returns the path to the data folder.
        '''
        particle_dict = {0: "photon",
                         1: "electron",
                         2: "pion"}
        
        path = "../data/" + particle_dict[self.particle_type] + "/"
        return path
    
    def load_data(self, type='train'):
        '''
        Loads the data from the data folder.
        Args:
            - type: str, 'train', 'valid', or 'test'
        Returns:
            - X: np.array, shape=(n_events, 51, 51)
            - y: np.array, shape=(n_events, n_pcl, 2)
            - en: np.array, shape=(n_events, n_pcl)
        '''
        self.path = self.data_path()

        X  = np.load(self.path + "X{}.npy".format(type))
        y  = np.load(self.path + "y{}.npy".format(type))
        en = np.load(self.path + "en{}.npy".format(type))

        return X, y, en
    
class Photon(Particle):
    '''
    Class for photon data.
    
    Args: 
        - self.n_pcl: int, number of particles in the event
        - self.path: str, path to the data folder
    '''

    def __init__(self, n_pcl=1) -> None:
        super().__init__(particle_type=0)
        self.n_pcl = n_pcl
    
    def data_path(self):
        '''
        Returns the path to the data folder.
        '''
        if self.n_pcl == 1:
            path = "../data/photon/" + "one_particle/"
        else:
            path = "../data/photon/" + "two_particles/"
        return path