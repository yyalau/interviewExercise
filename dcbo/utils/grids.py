from data_struct import hDict
import numpy as np
import tensorflow_probability as tfp

class DistU:
    def __init__(self, dist):
        self.dist = dist
    
    def sample(self, n_samples = 1):
        return self.dist.sample(n_samples).numpy()


def get_interv_sampler(exp_set, limits):
    
    i_sampler = hDict(variables = exp_set, nT = 1, nTrials = 1, )
    
    f= lambda i, es: np.array([limits[var][i] for var in es], dtype=np.float32)
    for es in exp_set:        
        low = f(0, es); high = f(1, es)
        i_sampler[es] = DistU(tfp.distributions.Uniform(low=low, high=high))

    return i_sampler