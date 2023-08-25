

class particle():
    '''
    Class for particle data.
    
    Args: 
        - self.particle_type: int, 0 for photon, 1 for electron, 2 for pion
        - self.n_pcl: int, number of particles in the event
        - self.path: str, path to the data folder
    '''

    def __init__(self, particle_type, n_pcl=1) -> None:
        self.particle_type = particle_type
        self.n_pcl = n_pcl

        self.path = self.data_path()
        
    def data_path(self):
        '''
        Returns the path to the data folder.
        '''
        particle_dict = {0: "photon", 
                         1: "electron", 
                         2: "pion"}
        
        path = "data/" + particle_dict[self.particle_type] + "/"
        
        # for photon additionally specify if there is one or two particles
        if self.particle_type == 0: 
            if self.n_pcl == 1: 
                path = path + "one_particle/"
            else: 
                path = path + "two_particles/"
        return path