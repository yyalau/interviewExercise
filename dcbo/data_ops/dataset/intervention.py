# from ..base import DatasetBase
from data_struct import esDict,hDict
import numpy as np

class DatasetInv:
    # TODO: remove nT and n_samples, because they should be derived from the self.dataX[es][t] / self.dataY[es][t]
    def __init__(
        self,
        exp_sets,
        nT,
        nTrials,
        dtype = "float32",
    ):
        self.n_samples = hDict(variables = exp_sets, nT = nT, nTrials = 1, default = lambda x,y : np.zeros((x,y), dtype = "int"), dtype = dtype)
        
        self.dataX = esDict(exp_sets, nT = nT, nTrials = nTrials, dtype = dtype)
        self.dataY = hDict(variables = exp_sets, nT = nT, nTrials = nTrials, dtype = dtype)
        self.dtype = dtype
    
    def update(self, es, t, *, x, y):
        '''
        x = trial_lvl = array([-2.294363021850586], dtype=object)
        y = y_new = -1.2014532
        '''
        
        idx = self.n_samples[es][t, 0]
        self.dataX[es][t, idx] = x
        self.dataY[es][t, idx] = y
                
        
        self.n_samples[es][t] += 1
    
    def get(self, es, t):
        
        idx = self.n_samples[es][t,0]
        return self.dataX[es][t, :idx].astype(self.dtype), self.dataY[es][t, :idx].astype(self.dtype)
    
    def __repr__(self):
        return f"DatasetInv(n_samples={self.n_samples})\n{self.dataX}\n{self.dataY}"        
