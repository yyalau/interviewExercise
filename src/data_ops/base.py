from data_struct import hDict
import numpy as np
from typing import Union, List
from sems import SEMBase

class DatasetBase:

    def __init__(self, nT: int, n_samples: Union[hDict, int], variables: Union[List, np.array], dtype="float32"):
        '''
        Initializes the DatasetBase object with the given data.
        Parameters:
        -----------
        nT : int
            The number of time points.
        n_samples : Union[hDict, int]
            The number of samples.
        variables : Union[List, np.array]
            The variables.
        dtype : str
            The data type.
        '''
        
        assert isinstance(n_samples, (int, hDict)), "n_samples must be an integer or an instance of hDict"
        assert isinstance(nT, int) and nT > 0, "the number of timesteps (n_samples) must be an integer greater than 0"

        self.n_samples = n_samples
        self.nT = nT
        self.variables = variables
        self.dtype = dtype

    def update(self):
        raise NotImplementedError("This method must be implemented in a subclass")

class DataSamplerBase:

    def __init__(self, sem: SEMBase, nT: int, variables: Union[List, np.array], dtype="float32"):
        '''
        Initializes the DataSamplerBase object with the given SEM and variables.
        Parameters:
        -----------
        sem : SEMBase
            The SEM object.
        nT : int
            The number of time points.
        variables : Union[List, np.array]
            The variables. 
        dtype : str
            The data type.
        '''
            
        assert isinstance(sem, SEMBase), "sem must be an instance of SEMBase"
        assert nT > 0 and isinstance(nT, int), "the number of timesteps (n_samples) must be an integer greater than 0"
        assert dtype in ['float32', 'float64'], "dtype must be one of 'float32' or 'float64"
        self.sem = sem
        self.nT = nT
        self.variables = variables
        self.dtype = dtype

    def sample(self):
        raise NotImplementedError("This method must be implemented in a subclass")
    