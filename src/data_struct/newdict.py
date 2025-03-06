from collections import OrderedDict
from typing import Union, Tuple, List, Any, Callable, Sequence
import numpy as np
import tensorflow as tf
import copy

class newDict(OrderedDict):
    def __init__(self, arr, dtype="float32") -> None:
        OrderedDict.__init__(self, arr)
        self.arr = arr
        self.dtype = dtype

    def __eq__(self, value):
        return super().__eq__(value) and self.__dict__ == value.__dict__

    def __deepcopy__(self, memo):

        # Create a new instance
        cls = self.__class__
        result = cls.__new__(cls)

        # Don't copy self reference
        memo[id(self)] = result
        
        result = cls(**{k: copy.deepcopy(v, memo) for k, v in self.__dict__.items()})
        
        # Return updated instance
        return result

    def add(self, key, *, t=0, trial=0, value=None):
        if self.get(key) is not None:
            return

        self[key] = self.default(self.nT, self.nTrials)

    def duplicate(self, nTrials):
        self.nTrials = self.nTrials * nTrials
        for k, v in self.items():
            self[k] = tf.repeat(v, nTrials, axis=-1)


    
class hDict(newDict):
    # graph > time > trials
    def __init__(
        self,
        *,
        variables: list[Tuple],
        nT: int = 1,
        nTrials: int = 1,
        default: Callable = None,
        dtype = "float32",
        arr = None
    ) -> None:
        self.variables = variables
        self.nT = nT
        self.nTrials = nTrials

        self.default = (
            default
            if default is not None
            else (lambda nT, nTrials: np.array([[None] * nTrials] * nT))
        )

        super().__init__(arr = [(es, self.default(nT, nTrials)) for es in variables] if arr is None else arr, dtype=dtype)


class esDict(newDict):

    def __init__(self, exp_sets, nT=1, nTrials=1, dtype="float32"):
        self.exp_sets = exp_sets
        self.nT = 1
        self.nTrials = 1

        self.default = lambda nT, nTrials, es_l: np.array(
            [[[None] * es_l] * nTrials] * nT
        )
        super().__init__(arr = [(es, self.default(nT, nTrials, len(es))) for es in exp_sets], dtype=dtype)


