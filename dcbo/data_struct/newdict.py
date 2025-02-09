from collections import OrderedDict
from typing import Union, Tuple, List, Any, Callable
import numpy as np


class newDict:
    def __init__(self, dtype="float32") -> None:
        self.dtype = dtype
    
    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value) -> None:
        self.data[key] = value

    def __delitem__(self, key) -> None:
        del self.data[key]

    def __iter__(self) -> iter:
        return self.data.__iter__()

    def __str__(self) -> str:
        return self.data.__str__()

    def __repr__(self):
        return self.data.__repr__()

    def get(self, key) -> Union[OrderedDict, None]:
        return self.data.get(key)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def add(self, key, *, t=0, trial=0, value=None):

        if self.data.get(key) is not None:
            return

        self.data[key] = self.default(self.nT, self.nTrials)

    def update(self, *others):
        for other in others:
            assert isinstance(
                other, (dict, newDict)
            ), "newd must be a dictionary or hDict"
            self.data.update(other)

    def duplicate(self, nTrials):
        self.nTrials = self.nTrials * nTrials
        for k, v in self.data.items():
            self.data[k] = v.repeat(nTrials, axis=-1)


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
    ) -> None:
        super().__init__(dtype=dtype)
        self.variables = variables
        self.nT = nT
        self.nTrials = nTrials

        self.default = (
            default
            if default is not None
            else (lambda nT, nTrials: np.array([[None] * nTrials] * nT))
        )

        self.data = OrderedDict([(es, self.default(nT, nTrials)) for es in variables])


class esDict(newDict):

    def __init__(self, exp_sets, nT=1, nTrials=1, dtype="float32"):
        super().__init__(dtype=dtype)
        self.exp_sets = exp_sets
        self.nT = 1
        self.nTrials = 1

        self.default = lambda nT, nTrials, es_l: np.array(
            [[[None] * es_l] * nTrials] * nT
        )

        self.data = {es: (self.default(nT, nTrials, len(es))) for es in exp_sets}

