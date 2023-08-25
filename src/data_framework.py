import numpy as np 
from sklearn.utils import shuffle

from network_utils.seed_finder import SeedNet
from network_utils.center_finder import CenterNet
from network_utils.analysis import NetAnalysis

class BaseNet(): 
    '''
    Base class to define the network and data samples. 
    
    Args: 
        - mode: 1 cluster, 2 cluster or 0 - mixed.
        - training: training if True, otherwise evaluation.
        - datapath: path to the data files. 
        - net: the network to be used - either seed or center. 
    '''
    def __init__(self, mode, datapath, training=True, net=None, datapath1=None, seed_path=None, 
                 seed_weight=None, center_path=None, center_weight=None, energy=False, pfclustering=False, 
                 thr=0.5, load=False): 
        
        self.mode = mode
        self.datapath = datapath
        self.training = training
        self.net = net
        self.energy = energy
        self.pf = pfclustering
        self.thr = thr
        self.load = load
        
        if self.mode == 0: 
            self.datapath = [datapath, datapath1]
            
        self.seed_path = seed_path
        self.seed_weight = seed_weight

        self.center_path = center_path 
        self.center_weight = center_weight
            
    def get_samples_network(self):
        '''
        Prepare samples and the network.
        '''
        if self.mode == 0: 
            TRAIN, TEST = self.load_mixed()
        
        else: 
            TRAIN, TEST = self.load_samples(path=self.datapath)
            
        if self.net == 'seed': 
            seed_net = SeedNet(self.mode, TEST, TRAIN)
            (Xtrain, ytrain), (Xtest, ytest) = seed_net.prepare_sample()
            model = seed_net.net_architecture()
            
            return (Xtrain, ytrain), (Xtest, ytest), model
            
        if self.net == 'center': 
            center_net = CenterNet(self.mode, TEST, TRAIN, energy=self.energy) 
            (Xtrain, ytrain, entrain, indtrain), (Xtest, ytest, entest, indtest) = center_net.prepare_sample()
            model = center_net.net_architecture()
            
            return (Xtrain, ytrain, entrain, indtrain), (Xtest, ytest, entest, indtest), model
        
    def get_analysis(self, test_samples=None, message_passing=False, one_neighbor=True): 
        '''
        Get the distribution for the full analysis.
        '''
     
        if not test_samples: 
            TEST = self.load_samples(path=self.datapath)
            
        else: # to be able to perform analysis on different networks with the same noise.
            TEST = test_samples

        analysis = NetAnalysis(TEST, n_cl=TEST[1].shape[1], 
                               seed_path=self.seed_path, seed_weight=self.seed_weight, 
                               center_path=self.center_path, center_weight=self.center_weight, 
                               enrg=self.energy, pfclustering=self.pf, message=message_passing, 
                               thr=self.thr, load=self.load, one_neighbor=one_neighbor)
        
        df = analysis.analysis_full()
        return df 
#         if self.pf: 
#             model, pfclustering, en_test = analysis.analysis_full()
#             return model, pfclustering, en_test
        
#         else:
#             model_dx, model_dy, model_dr, model_seed, model_enev, model_encr, en_test = analysis.analysis_full()
        
          #  return model_dx, model_dy, model_dr, model_seed, model_enev, model_encr, en_test
    
    def get_analysis_seed(self): 
        '''
        Get the distributions to perform the analysis of the seeding network.
        '''
        if self.training: 
            TRAIN, TEST = self.load_mixed()
            
            Xtrain, ytrain, entrain = TRAIN
            Xtest, ytest, entest = TEST
            
            Xcrop_train, ycrop_train, encrop_train, is_seed_train, ypr_train, indices_train = [], [], [], [], [], []
            Xcrop_test, ycrop_test, encrop_test, is_seed_test, ypr_test, indices_test = [], [], [], [], [], []
            
            for (X, y, en, n_cl) in zip(Xtrain, ytrain, entrain, (ytrain[0].shape[1], ytrain[1].shape[1])):
                print(X.shape, y.shape, en.shape)
                analysis_train = NetAnalysis((X, y, en, en), seed_path=self.seed_path, 
                                             seed_weight=self.seed_weight, n_cl=n_cl)
                
                Xcrop, ycrop, encrop, is_seed, ypr_seed, indices, n_seeds = analysis_train.analysis_seed()
                
                Xcrop_train.append(Xcrop)
                ycrop_train.append(ycrop)
                encrop_train.append(encrop)
                is_seed_train.append(is_seed)
                ypr_train.append(ypr_seed) 
                indices_train.append(indices)
                
            for (X, y, en, n_cl) in zip(Xtest, ytest, entest, (ytest[0].shape[1], ytest[1].shape[1])):
                analysis_test = NetAnalysis((X, y, en, en), seed_path=self.seed_path, 
                                 seed_weight=self.seed_weight, n_cl=n_cl)
                
                Xcrop, ycrop, encrop, is_seed, ypr_seed, indices, _ = analysis_test.analysis_seed()
                Xcrop_test.append(Xcrop)
                ycrop_test.append(ycrop)
                encrop_test.append(encrop)
                is_seed_test.append(is_seed)
                ypr_test.append(ypr_seed) 
                indices_test.append(indices)
                
            Xcrop_train = np.concatenate(Xcrop_train, axis=0)
            ycrop_train = np.concatenate(ycrop_train, axis=0)
            encrop_train = np.concatenate(encrop_train, axis=0)
            is_seed_train = np.concatenate(is_seed_train, axis=0)
            ypr_train = np.concatenate(ypr_train, axis=0)
            indices_train = np.concatenate(indices_train, axis=0)
            
            Xcrop_test = np.concatenate(Xcrop_test, axis=0)
            ycrop_test = np.concatenate(ycrop_test, axis=0)
            encrop_test = np.concatenate(encrop_test, axis=0)
            is_seed_test = np.concatenate(is_seed_test, axis=0)
            ypr_test = np.concatenate(ypr_test, axis=0)
            indices_test = np.concatenate(indices_test, axis=0)
            
            Xcrop_train, ycrop_train, encrop_train, is_seed_train, ypr_train, indices_train = shuffle(Xcrop_train, ycrop_train, encrop_train, is_seed_train, ypr_train, indices_train, random_state=42)
            Xcrop_test, ycrop_test, encrop_test, is_seed_test, ypr_test, indices_test = shuffle(Xcrop_test, ycrop_test, encrop_test, is_seed_test, ypr_test, indices_test, random_state=42)
            
            return (Xcrop_train, ycrop_train, encrop_train, is_seed_train, ypr_train, indices_train), (Xcrop_test, ycrop_test, encrop_test, is_seed_test, ypr_test, indices_test)
        
        else:
            (Xtest, ytest, entest, _) = self.load_samples(path=self.datapath)
            analysis = NetAnalysis((Xtest, ytest, entest, entest), seed_path=self.seed_path, 
                                   seed_weight=self.seed_weight, n_cl=self.mode)

            X_crop, y_crop, en_crop, is_seed, ypr_seed, indices, n_seeds = analysis.analysis_seed()
            
            return X_crop, y_crop, en_crop, is_seed, ypr_seed, indices
            
    def load_mixed(self): 
        '''
        Function to load the mixed sample. 
        '''
        Xtrain, ytrain, entrain = [], [], []
        Xtest, ytest, entest = [], [], []

        for path in self.datapath: 
            (X1, yc1, en1, _), (X2, yc2, en2, _) = self.load_samples(path=path)
            if len(yc1.shape) != 3:
                yc1 = np.expand_dims(yc1, axis=1) 
                yc2 = np.expand_dims(yc2, axis=1) 

            Xtrain.append(X1)
            Xtest.append(X2)

            ytrain.append(yc1) 
            ytest.append(yc2) 

            entrain.append(en1)
            entest.append(en2) 
        return (Xtrain, ytrain, entrain), (Xtest, ytest, entest)
      
    def load_samples(self, path, training=True): 
        '''
        Upload samples for training and evaluation. 
        
        Args: 
            - training: if True upload training and validation samples, 
                        else only testing sample
        '''
        
        if self.training: 
            Xtrain = np.load(path+'Xtrain.npy')
            Xtest = np.load(path+'Xvalid.npy')

            yctrain = np.load(path+'ytrain.npy')
            yctest = np.load(path+'yvalid.npy')

            entrain = np.load(path+'entrain.npy')
            entest = np.load(path+'envalid.npy')
            
            if len(yctrain.shape) != 3:
                yctrain = np.expand_dims(yctrain, axis=1) 
                yctest = np.expand_dims(yctest, axis=1) 
                
            edeptrain = np.sum(Xtrain, axis=(1,2))
            edeptest = np.sum(Xtest, axis=(1,2))
            
            Xtrain = self.apply_noise(Xtrain)
            Xtest = self.apply_noise(Xtest)
            print(Xtrain.shape, Xtest.shape)
            
            return (Xtrain, yctrain, entrain, edeptrain), (Xtest, yctest, entest, edeptest) 
            
        else: 
            Xtest = np.load(self.datapath+'Xtest.npy')
            yctest = np.load(self.datapath+'ytest.npy')
            entest = np.load(self.datapath+'entest.npy')
            print(Xtest.shape, entest.shape)
            edeptest = np.sum(Xtest, axis=(-1,-2))
            
            if len(yctest.shape) != 3:
                yctest = np.expand_dims(yctest, axis=1) 
            Xtest = self.apply_noise(Xtest)
            
            return (Xtest, yctest, entest, edeptest)
        
    def apply_noise(self, X): 
        '''
        Apply noise on the sample (167 MeV) and cut at 50 MeV. 
        '''
        n = 0.167
        s = 0.03
        c = 0.7 * 0.005
        
        #sigma = np.sqrt(X*s**2 + n**2 + X**2 * c**2)
        #print(sigma.max(), sigma.min())
        sigma = 0.167
        X = np.random.normal(X, sigma)
        
        #X = np.random.normal(X, 0.167)
        
        X[X < 0.05] = 0.
        return X 
    
