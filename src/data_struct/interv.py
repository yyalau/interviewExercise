from collections import defaultdict
from .newdict import esDict
import numpy as np
from copy import deepcopy

class IntervLog:
    
    
    def __init__(self, exp_sets, nT = 1, nTrials = 1):
        # along time
        '''
        at each time t
            for each trial
                for each exploration set
                    the intervention set and level
        '''

        self.n_keys = 4
        self.keys = {
            "impv": 0,
            "i_set": 1,
            "i_level": 2,
            "y_values": 3,
            "idx": 4,
        }
        self.data = np.array([[[[None] * self.n_keys] * len(exp_sets)] * nTrials] * nT)
        self.nT = nT
        self.nTrials = nTrials
        
        self.set_idx = {es: idx for idx, es in enumerate(exp_sets)}
        '''
        opt_data 1
        
        at each time t
            (tuple) the optimal es set and intervention level
            
        opt_data 2
        at each time t
            for each trial
                (tuple) the optimal es set and intervention level
        '''
    def get_value(self, x, key = 0, direction = max):
        
        temp = [v for v in x if v[key] is not None]
        if len(temp) == 0:
            return None        
        return direction(temp, key=lambda x: x[key])
    

    @property
    def sol(self):
        r = np.array([[None]*self.n_keys]* self.nT)        
        for t in range(self.nT):
            r[t] = self.get_value(self.opt_y_trial[t], key = self.keys["y_values"], direction = min)
        return r
    
    @property
    def opt_impv_trial(self):

        r = np.array([[[None]*self.n_keys]* self.nTrials]* self.nT )
        
        for t in range(self.nT):
            for trial in range(self.nTrials):
                # r[t, trial] = self.get_opt_ptrial(t, trial)
                r[t, trial] = self.get_value(self.data[t, trial], key = self.keys["impv"], direction = max)
        return r

    @property
    def opt_y_trial(self):
        r = np.array([[[None]*self.n_keys]* self.nTrials]* self.nT )
        
        for t in range(self.nT):
            for trial in range(self.nTrials):
                # r[t, trial] = self.get_opt_ptrial(t, trial)
                r[t, trial] = self.get_value(self.data[t, trial], key = self.keys["y_values"], direction = min)

        return r

    
    def update_y(self, t, trial, es, y_values):
        self.data[t, trial, self.set_idx[es]][self.keys["y_values"]] = y_values
                        

    def update(self, t, trial, *, impv,  y_values, i_set, i_level):
        self.data[t,trial, self.set_idx[i_set]] = np.array([impv, i_set, i_level, y_values], dtype = object)
    
    
    def __repr__(self):
        return self.data.__repr__()
if __name__ == "__main__":
    nT = 4; nTrials = 5
    il = IntervLog( exp_sets= ('X','Z', 'XZ'), nT = nT, nTrials = nTrials)
    il.update(0, 0, impv = 10, y_values = 100, i_set = "X", i_level = 1)
    il.update(0, 0, impv = 20, y_values = 200, i_set = "Z", i_level = 2)
    il.update(0, 2, impv = 30, y_values = 300, i_set = "X", i_level = 3)
    il.update(0, 3, impv = 40, y_values = 400, i_set = "Z", i_level = 4)
    il.update(0, 3, impv = 30, y_values = 500, i_set = "X", i_level = 5)
    il.update(1, 3, impv = 40, y_values = 600, i_set = "Z", i_level = 6)
    il.update(1, 3, impv = 30, y_values = 700, i_set = "X", i_level = 7)
    il.update(1, 3, impv = 40, y_values = 800, i_set = "Z", i_level = 8)

    # print(il.data[0])
    # print(il.opt_ptrial[0])
    # print(il.sol[0])    
    
