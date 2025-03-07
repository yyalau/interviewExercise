# from ..base import DatasetBase
from data_struct import esDict, hDict
import numpy as np
import tensorflow as tf
from ..base import DatasetBase
from typing import Union, List, Tuple


class DatasetInv(DatasetBase):
    def __init__(
        self,
        exp_sets: Union[List, np.array],
        nT: int,
        nTrials: int,
        dtype: str ="float32",
    ) -> None:
        '''
        Initializes the DatasetInv object with the given data.
        Parameters:
        -----------
        exp_sets : Union[List, np.array]
            The experimental sets.
        nT : int
            The number of time points.
        nTrials : int
            The number of trials.
        dtype : str
            The data type.
        '''
        
        self.sc(exp_sets)
        super().__init__(nT, 1, exp_sets, dtype)

        self.n_samples = hDict(
            variables=exp_sets,
            nT=nT,  
            nTrials=1,
            default=lambda x, y: np.zeros((x, y), dtype="int"),
            dtype=dtype,
        )

        self.dataX = esDict(exp_sets, nT=nT, nTrials=nTrials, dtype=dtype)
        self.dataY = hDict(variables=exp_sets, nT=nT, nTrials=nTrials, dtype=dtype)
    

    def update(self, es: Tuple, t: int, *, x: np.array, y: float):
        """
        Updates the dataset with the given data.
        Parameters:
        -----------
        es : Tuple
            The experimental set.
        t : int
            The time point.
        x : np.array
            The input data.
        y : float
            The output data.
        """

        self.sc_update(es, t, x, y)    
        
        idx = self.n_samples[es][t, 0]
        self.dataX[es][t, idx] = x
        self.dataY[es][t, idx] = y
        self.n_samples[es][t] += 1

    def get_tf(self, es: Tuple, t: int) -> Tuple[tf.Tensor, tf.Tensor]:

        idx = self.n_samples[es][t, 0]
        x = tf.convert_to_tensor(self.dataX[es][t, :idx].astype(self.dtype))
        y = tf.reshape(tf.convert_to_tensor(self.dataY[es][t, :idx].astype(self.dtype)), (-1,))
        return x,y

    def __repr__(self):
        return f"DatasetInv(n_samples={self.n_samples})\n{self.dataX}\n{self.dataY}"

    def sc(self, exp_sets):
        
        assert isinstance(exp_sets, (List, Tuple, np.array)), "exp_sets must be a list or an array"
        assert len(exp_sets) > 0, "exp_sets must have at least one element"
        for es in exp_sets:
            assert isinstance(es, tuple), f"exp_set must be a tuple, got {type(es)}"
    
    def sc_update(self, es, t, x, y):
        assert isinstance(es, Tuple) and es in self.variables, f"Invalid es: {es}"
        assert isinstance(t, int) and t >= 0 and t < self.nT, f"t must be an integer between 0 and {self.nT-1}, got {t}"
        assert isinstance(x, np.ndarray), f"x must be a np.ndarray, got {type(x)}"
        assert x.shape[0] == len(es), f"x must have shape ({len(es)},), got {x.shape}"
        assert isinstance(y, (float, np.float_, np.float32, np.float64)), f"y must be a float, got {type(y)}"
