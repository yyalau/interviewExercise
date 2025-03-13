from collections import OrderedDict
from typing import Union, Tuple, List, Any, Callable, Sequence, Dict
from .node import Var
import numpy as np
import tensorflow as tf
import copy

class newDict(OrderedDict):
    def __init__(self, arr: List[Tuple[Any, Any]], nT, nTrials) -> None:
        '''
        Parameters:
        -----------
        arr: List[Tuple[Any, Any]]
            List of tuples containing the key and value.
        nT: int
            Number of time indices.
        nTrials: int
            Number of trials.            
        '''
        
        self.sc(arr, nT, nTrials)
        
        OrderedDict.__init__(self, arr)
        self.nT = nT
        self.nTrials = nTrials
        self.arr = arr
    
    def sc(self, arr, nT, nTrials):
        
        for i in arr:
            assert isinstance(i, tuple), f"Expected tuple in format (key, value), got {type(i)}"
            assert len(i) == 2, f"Expected tuple of length 2 in format (key, value), got {len(i)}"
            assert isinstance(i[0], (Var, tuple)), f"Expected key to be of type Var or tuple, got {type(i[0])}"
            if isinstance(i[0], tuple):
                assert len(i[0]) > 0, f"Expected tuple of length > 0, got {len(i[0])}"
            
            # if i[1] is not None:
            #     assert isinstance(i[1], (np.ndarray, tf.Tensor)), f"Expected value to be of type np.ndarray or tf.Tensor, got {type(i[1])}"
            # assert type(arr[0][1]) == type(i[1]), f"Expected all values to be of the same type {type(arr[0][1])}, got {type(i[1])}"
        
        
        assert isinstance(nT, int) and nT > 0, f"Expected nT to be a positive integer, got {nT}"
        assert isinstance(nTrials, int) and nTrials > 0, f"Expected nTrials to be a positive integer, got {nTrials}"

    def __deepcopy__(self, memo: Dict[int, Any]) -> 'newDict':
        # Create a new instance
        cls = self.__class__
        result = cls.__new__(cls)

        # Don't copy self reference
        memo[id(self)] = result
        result = cls([k for k in self.keys()], **{k: copy.deepcopy(v, memo) for k, v in self.__dict__.items()})
        
        # Return updated instance
        return result

    def add(self, key: Union[Var, Tuple[Var]]) -> None:
        '''
        Parameters:
        -----------
        key: Any
            Key of the dictionary.
        t: int
            Time index.
        trial: int
            Trial index.
        value: Any
            Value to be added.
        '''
        
        assert isinstance(key, (Var, tuple)), f"Expected key to be of type Var or tuple, got {type(key)}"
        if isinstance(key, tuple):
            assert len(key) > 0, f"Expected tuple of length > 0, got {len(key)}"
        if self.get(key) is not None:
            return

        self[key] = self.default(self.nT, self.nTrials)

    def duplicate(self, nTrials: int) -> None:
        '''
        Duplicates the values of the dictionary by the number of trials.
        Parameters:
        -----------
        nTrials: int
            Number of trials.
        '''
        self.nTrials = self.nTrials * nTrials
        for k, v in self.items():
            self[k] = tf.repeat(v, nTrials, axis=-1)


class hDict(newDict):
    # graph > time > trials
    def __init__(
        self,
        variables: List[Var],
        nT: int = 1,
        nTrials: int = 1,
        default: Callable[[int, int], np.ndarray] = None,
        arr: List[Tuple[Any, Any]] = None
    ) -> None:
        '''
        Parameters:
        -----------
        variables: List[Var]
            List of tuples containing the key and value.
        nT: int
            Number of time indices.
        nTrials: int
            Number of trials.
        default: Callable[[int, int], np.ndarray]
            Default value.
        arr: List[Tuple[Any, Any]]
            List of tuples containing the key and value.
        '''
        self.sc_hdict(variables)

        self.default = (
            default
            if default is not None
            else (lambda nT, nTrials: np.array([[None] * nTrials] * nT))
        )

        super().__init__(arr = [(es, self.default(nT, nTrials)) for es in variables] if arr is None else arr, nT = nT, nTrials = nTrials)

    def sc_hdict(self, variables):
        assert isinstance(variables, (list, np.ndarray)), "variables must be a list or an array"

class esDict(newDict):

    def __init__(self, exp_sets: List[Tuple[Var]], nT: int = 1, nTrials: int = 1) -> None:
        '''
        Parameters:
        -----------
        exp_sets: List[Tuple[Var]]
            List of tuples containing the variables
        nT: int
            Number of time indices.
        nTrials: int
            Number of trials.
        '''
        self.sc_esdict(exp_sets)

        self.default = lambda nT, nTrials, es_l,: np.array(
            [[[None] * es_l] * nTrials] * nT
        )
        super().__init__(arr = [(es, self.default(nT, nTrials, len(es), )) for es in exp_sets], nT = nT, nTrials = nTrials)

    def sc_esdict(self, exp_sets):
        assert isinstance(exp_sets, (list, np.ndarray)), "exp_sets must be a list or an array"
        assert len(exp_sets) > 0, "exp_sets must have at least one element"
        for es in exp_sets:
            assert isinstance(es, tuple), f"exp_set must be a tuple, got {type(es)}"
            assert len(es) > 0, f"exp_set must have at least one element"
            for v in es:
                assert isinstance(v, Var), f"exp_set must contain only Var, got {type(v)}"