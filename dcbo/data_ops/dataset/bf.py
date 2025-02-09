import numpy as np


class DatasetBF:
    def __init__(self, n_samples=0, epsilon=None, dataX = None, dataY=None, dtype="float32"):
        # super().__init__(1, n_samples, dataY)
        self.dataX = dataX
        self.dataY = dataY
        self.epsilon = epsilon
        self.n_samples = n_samples
        self.dtype = dtype
        
    def update_new(self, x, y):
        # TODO: verify correctness
        
        if self.dataX is None and self.dataY is None:
            self.dataX = x
            self.dataY = y
            return
        
        self.dataX = np.concatenate([self.dataX, x])
        self.dataY = np.concatenate([self.dataY, y])
        
        if self.dtype is None:
            self.dtype = y.dtype
        