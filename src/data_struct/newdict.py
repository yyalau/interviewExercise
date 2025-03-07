from collections import OrderedDict
from typing import Union, Tuple, List, Any, Callable, Sequence, Dict
from .node import Var
import numpy as np
import tensorflow as tf
import copy

class newDict(OrderedDict):
    def __init__(self, arr: List[Tuple[Any, Any]], dtype: str = "float32") -> None:
        '''
        Parameters:
        -----------
        arr: List[Tuple[Any, Any]]
            List of tuples containing the key and value.
        dtype: str
            Data type of the values.
        '''
        OrderedDict.__init__(self, arr)
        self.arr = arr
        self.dtype = dtype

    def __eq__(self, value: Any) -> bool:
        return super().__eq__(value) and self.__dict__ == value.__dict__

    def __deepcopy__(self, memo: Dict[int, Any]) -> 'newDict':
        # Create a new instance
        cls = self.__class__
        result = cls.__new__(cls)

        # Don't copy self reference
        memo[id(self)] = result
        
        result = cls(**{k: copy.deepcopy(v, memo) for k, v in self.__dict__.items()})
        
        # Return updated instance
        return result

    def add(self, key: Any, *, t: int = 0, trial: int = 0, value: Any = None) -> None:
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
        if self.get(key) is not None:
            return

        self[key] = self.default(self.nT, self.nTrials, self.dtype)

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
        *,
        variables: List[Var],
        nT: int = 1,
        nTrials: int = 1,
        default: Callable[[int, int], np.ndarray] = None,
        dtype: str = "float32",
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
        dtype: str
            Data type of the values.
        arr: List[Tuple[Any, Any]]
            List of tuples containing the key and value.
        '''
        self.variables = variables
        self.nT = nT
        self.nTrials = nTrials

        self.default = (
            default
            if default is not None
            else (lambda nT, nTrials, dtype: np.array([[None] * nTrials] * nT))
        )

        super().__init__(arr = [(es, self.default(nT, nTrials, dtype)) for es in variables] if arr is None else arr, dtype=dtype)


class esDict(newDict):

    def __init__(self, exp_sets: List[Tuple[Var]], nT: int = 1, nTrials: int = 1, dtype: str = "float32") -> None:
        '''
        Parameters:
        -----------
        exp_sets: List[Tuple[Var]]
            List of tuples containing the variables
        nT: int
            Number of time indices.
        nTrials: int
            Number of trials.
        dtype: str
            Data type of the values.
        '''
        self.exp_sets = exp_sets
        self.nT = 1
        self.nTrials = 1

        self.default = lambda nT, nTrials, es_l, dtype: np.array(
            [[[None] * es_l] * nTrials] * nT
        )
        super().__init__(arr = [(es, self.default(nT, nTrials, len(es), dtype)) for es in exp_sets], )
