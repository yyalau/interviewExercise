import numpy as np
from typing import Optional

class DatasetBF:
    def __init__(self, dataX: Optional[np.array] = None, dataY: Optional[np.array]=None, dtype: str="float32") ->None:
        '''
        Initializes the DatasetBF object with the given data.
        Parameters:
        -----------
        dataX : Optional[np.array]
            The input data.
        dataY : Optional[np.array]
            The output data.
        dtype : str
            The data type.
        '''
        self.sc(dataX, dataY)

        self.dataX = dataX
        self.dataY = dataY
        self.n_samples = dataX.shape[0] if dataX is not None else None
        self.dtype = dtype
        
    def update(self, x: np.array, y: np.array) ->None:
        
        '''
        Updates the dataset with the given data.
        Parameters:
        -----------
        x : np.array
            The input data.
        y : np.array
            The output data.
        '''
        self.sc_update(x, y)
        
        if self.dataX is None and self.dataY is None:
            self.dataX = x
            self.dataY = y
            return
        
        self.dataX = np.concatenate([self.dataX, x])
        self.dataY = np.concatenate([self.dataY, y])        

    def sc(self, dataX, dataY):
        
        if dataX is None and dataY is None:
            return
        assert dataX is not None and dataY is not None, "dataX and dataY must not be None if one of them is not None"
        assert isinstance(dataX, np.ndarray), "dataX must be a numpy array"
        assert isinstance(dataY, np.ndarray), "dataY must be a numpy array"
        assert dataX.shape[0] == dataY.shape[0], "dataX and dataY must have the same number of samples"
        assert len(dataX.shape) == len(dataY.shape) == 2, "dataX and dataY must have 2 dimensions"
    
    def sc_update(self, x, y):
        assert isinstance(x, np.ndarray), "x must be a numpy array"
        assert isinstance(y, np.ndarray), "y must be a numpy array" 
        assert x.shape[0] == y.shape[0], "x and y must have the same number of samples"
        assert len(x.shape) == len(y.shape) == 2, "x and y must have 2 dimensions"
        
        if self.dataX is None and self.dataY is None: return
        assert x.shape[1] == self.dataX.shape[1], "x and dataX must have the same number of features"
        assert y.shape[1] == self.dataY.shape[1], "y and dataY must have the same number of features"
