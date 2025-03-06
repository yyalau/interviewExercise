
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np


class KernelDensity:
    def __init__(self, kernel, n_bins =100, bandwidth=1.0):        
        self.kernel = kernel
        self.n_bins = n_bins
        self.bandwidth = bandwidth
        
    
    def _set_bandwidth(self, x):    
        
        if isinstance(self.bandwidth, str):
            if self.bandwidth not in ["scott", "silverman"]:
                raise ValueError("Invalid bandwidth. Must be 'scott' or 'silverman'")
            
            if self.bandwidth == "scott":
                self.bandwidth_ = x.shape[0] ** (-1 / (x.shape[1] + 4))
            elif self.bandwidth == "silverman":
                self.bandwidth_ = (x.shape[0] * (x.shape[1] + 2) / 4) ** (-1 / (x.shape[1] + 4))
        else:
            self.bandwidth_ = self.bandwidth        
        
    def fit(self, x):
        self._set_bandwidth(x)
        
        
        # https://astroviking.github.io/ba-thesis/tensorflowImplementation.html
        probs = tf.ones(x.shape[0], dtype = x.dtype) / x.shape[0]
        f = lambda x: tfd.Independent(self.kernel(loc=x, scale=self.bandwidth_))         
        
        self.kde = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=probs),
            components_distribution=f(x),
            )
        
        return self
    
    def sample(self, n_samples):
        return self.kde.sample(n_samples)

if __name__ == "__main__":

    tf.random.set_seed(123)
    np.random.seed(123)

    X = np.random.rand(12,1)
    Y = np.random.rand(12)
    kde = KernelDensity(tfd.Normal).fit(X)
    
