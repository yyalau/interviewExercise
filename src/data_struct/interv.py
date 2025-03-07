from collections import defaultdict
from .newdict import esDict
import numpy as np
from copy import deepcopy
from . import Var
from typing import List, Tuple, Union, Dict, Any, Callable, Optional

class IntervLog:
    
    def __init__(self, exp_sets: List[Tuple[Var, ...]], nT: int = 1, nTrials: int = 1):
        '''
        Initializes the IntervLog object with the given exp_sets, nT, and nTrials.
        Parameters:
        -----------
        exp_sets : List[Tuple[Var, ...]]
            The exploration sets.
        nT : int
            The number of time steps.
        nTrials : int
            The number of trials.
        '''
        self.sc(nT, nTrials, exp_sets)

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

    def sc(self, nT: int, nTrials: int, exp_sets: Tuple[str, ...]) -> None:
        '''
        Checks if the given nT, nTrials, and exp_sets are valid.
        Parameters:
        -----------
        nT : int
            The number of time steps.
        nTrials : int
            The number of trials.
        exp_sets : Tuple[str, ...]
            The exploration sets.
        '''
        assert isinstance(nT, int) and nT > 0, "The number of time steps must be an integer greater than 0."
        assert isinstance(nTrials, int) and nTrials > 0, "The number of trials must be an integer greater than 0."
        assert len(exp_sets) > 0, "The exploration sets must not be empty."
        assert all(isinstance(es, tuple) and len(es) > 0 for es in exp_sets), "The exploration sets must be a tuple of tuples."
        assert all(isinstance(v, Var) for es in exp_sets for v in es), "The exploration sets must be a tuple of tuples of Var objects."
        
    def get_value(self, x: np.ndarray, key: int = 0, direction: Callable = max) -> Union[None, List[Any]]:
        '''
        Returns min / max of the value of the given key in the given array.
        Parameters:
        -----------
        x : np.array
            The array to search.
        key : int
            The key to search.
        direction : Callable
            The direction to search.
        Returns:
        --------
        Union[None, List[Any]]
            The min / max of the value of the given key in the given array.
        '''
        assert isinstance(x, np.ndarray), "The array must be a numpy array."
        assert isinstance(key, int) and key in [0,3], "The key must be either 0 (improvement) or 3 (y values)."
        assert direction in [min, max], "The direction must be either min or max (callable)."
        
        temp = [v for v in x if v[key] is not None]
        if len(temp) == 0:
            return None        
        return direction(temp, key=lambda x: x[key])
    

    @property
    def sol(self) -> np.ndarray:
        '''
        Returns the solution of the optimization algorithm.
        Returns:
        --------
        np.ndarray
            The solution of the optimization algorithm.
        '''
        r = np.array([[None]*self.n_keys]* self.nT)        
        for t in range(self.nT):
            r[t] = self.get_value(self.opt_y_trial[t], key = self.keys["y_values"], direction = min)
        return r
    
    @property
    def opt_impv_trial(self) -> np.ndarray:
        '''
        Returns the optimal improvement for each trial.
        Returns:
        --------
        np.ndarray
            The optimal improvement for each trial.
        '''
        r = np.array([[[None]*self.n_keys]* self.nTrials]* self.nT )
        
        for t in range(self.nT):
            for trial in range(self.nTrials):
                r[t, trial] = self.get_value(self.data[t, trial], key = self.keys["impv"], direction = max)
        return r

    @property
    def opt_y_trial(self) -> np.ndarray:
        '''
        Returns the optimal y values for each trial.
        Returns:
        --------
        np.ndarray
            The optimal y values for each trial.
        '''
        r = np.array([[[None]*self.n_keys]* self.nTrials]* self.nT )
        
        for t in range(self.nT):
            for trial in range(self.nTrials):
                r[t, trial] = self.get_value(self.data[t, trial], key = self.keys["y_values"], direction = min)

        return r

    
    def update_y(self, t: int, trial: int, es: Tuple[Var, ...], y_values: float) -> None:
        '''
        Updates the y values for the given time step, trial, and exploration set.
        Parameters:
        -----------
        t : int
            The time step.
        trial : int
            The trial.
        es : Tuple[Var, ...]
            The exploration set.
        y_values : float
            The y values.
        '''
        assert isinstance(t, int) and t >= 0 and t < self.nT, "The time step must be an integer between 0 and nT-1."
        assert isinstance(trial, int) and trial >= 0 and trial < self.nTrials, "The trial must be an integer between 0 and nTrials-1."
        assert isinstance(es, tuple), "The exploration set must be a tuple."
        assert len(es) > 0, "The exploration set must not be empty."
        assert all(isinstance(v, Var) for v in es), "The exploration set must be a tuple of Var objects."
        assert self.set_idx.get(es) is not None, "The exploration set is not valid."
        assert isinstance(y_values, (int, float, np.float32, np.float64)), "The y values must be an integer or a float."
        
        self.data[t, trial, self.set_idx[es]][self.keys["y_values"]] = y_values
                        

    def update(self, t: int, trial: int, *, impv: float, y_values: Optional[float], i_set: Tuple[Var,...], i_level: np.array) -> None:
        '''
        Updates the given time step, trial, improvement, y values, exploration set, and level.
        Parameters:
        -----------
        t : int
            The time step.
        trial : int
            The trial.
        impv : float
            The improvement from acquisition function.
        y_values : float
            The y values.
        i_set : Tuple[Var,...]
            The exploration set.
        i_level : np.array
            The level.
        '''
        assert isinstance(t, int) and t >= 0 and t < self.nT, "The time step must be an integer between 0 and nT-1"
        assert isinstance(trial, int) and trial >= 0 and trial < self.nTrials, "The trial must be an integer between 0 and nTrials-1."
        assert isinstance(impv, (int, float)), "The improvement must be an integer or a float."
        if y_values is not None:
            assert isinstance(y_values, (int, float, np.float32, np.float64)), "The y values must be an integer or a float."
        assert isinstance(i_set, tuple), "The exploration set must be a tuple."
        assert len(i_set) > 0, "The exploration set must not be empty."
        assert all(isinstance(v, Var) for v in i_set), "The exploration set must be a tuple of Var objects."        
        assert isinstance(i_level, np.ndarray), "The level must be a numpy array."
        assert self.set_idx.get(i_set) is not None, "The exploration set is not valid."
        assert i_level.shape[0] == len(i_set), "The level must have the same length as the exploration set."
        
        self.data[t,trial, self.set_idx[i_set]] = np.array([impv, i_set, i_level, y_values], dtype = object)
    
    
    def __repr__(self) -> str:
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
