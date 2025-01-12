# from ..base import DatasetBase
from data_struct import esDict
class DatasetInv:
    # TODO: remove nT and n_samples, because they should be derived from the dataX / dataY
    def __init__(
        self,
        nT: int,
        exp_sets
    ):
        self.n_samples = 0
        self.dataX = esDict(exp_sets,nT)
        self.dataY = esDict(exp_sets,nT)
        
        ''' In DatasetBase,
        self.dataX = {
            'initial_values': initial_values,
            'interv_levels': interv_levels,
            'epsilon': epsilon,
        }
        self.dataY = dataY
        '''
    
    def update(self, es, t, *, x, y):

        self.dataX[es][t, self.n_samples] = x
        self.dataY[es][t, self.n_samples] = y
        self.n_samples += 1
    
    def __repr__(self):
        return f"DatasetInv(n_samples={self.n_samples})\n{self.dataX}\n{self.dataY}"        
