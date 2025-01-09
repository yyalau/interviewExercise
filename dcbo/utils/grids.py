from data_struct import hDict
import numpy as np

def get_interv_grids(exp_set, limits, intervals):
    
    i_grids = hDict(variables = exp_set, nT = 1, nTrials = 1, )
    
    for es in exp_set:        
        extrema = np.vstack([limits[var] for var in es])
        inputs = [np.linspace(i, j, intervals) for i, j in extrema]
        i_grids[es] = np.dstack(np.meshgrid(*inputs)).ravel("F").reshape(len(inputs), -1).T        

    return i_grids