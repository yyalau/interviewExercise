# from ..base import DatasetBase
from data_struct import esDict,hDict
import numpy as np

class DatasetInv:
    # TODO: remove nT and n_samples, because they should be derived from the dataX / dataY
    def __init__(
        self,
        nT: int,
        exp_sets,
        dtype = "float32",
    ):
        self.n_samples = {es: 0 for es in exp_sets}
        self.dataX = esDict(exp_sets,nT)
        self.dataY = hDict(variables = exp_sets, nT = nT)
        self.dtype = dtype
    
    def update(self, es, t, *, x, y):
        
        dataX = self.dataX[es][t]; dataY = self.dataY[es][t]
        if self.n_samples[es] == 0:
            dataX[0] = x
            dataY[0] = y
        else:  
            f = lambda old, new: np.vstack([old, new])
            dataX = f(dataX, x) 
            dataY = f(dataY, y)
            
        self.n_samples[es] += 1
    
    def get(self, es, t):
        
        return self.dataX[es][t].astype(self.dtype), self.dataY[es][t].astype(self.dtype)
    
    def __repr__(self):
        return f"DatasetInv(n_samples={self.n_samples})\n{self.dataX}\n{self.dataY}"        
