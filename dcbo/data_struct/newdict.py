from collections import OrderedDict
from typing import Union, Tuple, List, Any, Callable
import numpy as np


class newDict:
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

    def reduce(self, t):
        self.nT = 1
        for k, v in self.data.items():
            self.data[k] = np.array([v[t]])


# TODO: not well structured
class gDict(newDict):
    # for graph
    def __init__(
        self, exp_set: List[Tuple], default: Callable = lambda: None, *args
    ) -> None:
        super().__init__()
        self.exp_set = exp_set
        self.data = OrderedDict([(es, default(*args)) for es in exp_set])


class tgDict(newDict):
    # for time > graph
    def __init__(
        self, exp_set: list[Tuple], nT: int, default: Callable = lambda: None, *args
    ) -> None:
        super().__init__()
        self.exp_set = exp_set
        self.nT = nT

        self.data = OrderedDict(
            [
                (t, OrderedDict([(e, default(*args)) for e in exp_set]))
                for t in range(nT)
            ]
        )


class hDict(newDict):
    # graph > time > trials
    def __init__(
        self,
        *,
        variables: list[Tuple],
        nT: int = 1,
        nTrials: int = 1,
        default: Callable = None,
    ) -> None:
        super().__init__()
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

    def __init__(self, exp_sets, nT=1, nTrials=1):
        super().__init__()
        self.exp_sets = exp_sets
        self.nT = 1
        self.nTrials = 1

        self.default = lambda nT, nTrials, es_l: np.array(
            [[[None] * es_l] * nTrials] * nT
        )

        self.data = {es: (self.default(nT, nTrials, len(es))) for es in exp_sets}


# test add and update function
# if __name__ == "__main__":
#     A = hDict(variables=[], nT=4, nTrials=1)
#     print(A)
#     A.add("X")

#     A.update({"Y": np.random.randn(4, 1)})
#     print(A)
#     A.add("X")
#     print(A)


# TODO: learn hashing
# def get(self, *, variable=None, t=None, trial=None):

#     if (variable, t, trial) == (None, None, None):
#         return self.data

#     if (t, trial) == (None, None):
#         return self.data[variable]

#     if (variable, trial) == (None, None):
#         return {k: v[t] for k, v in self.data.items()}

#     if (variable, t) == (None, None):
#         return {k: v[:, trial] for k, v in self.data.items()}

#     if trial == None:
#         return self.data[variable][t]

#     if t == None:
#         return self.data[variable][:, trial]

#     if variable == None:
#         return {k: v[t, trial] for k, v in self.data.items()}

#     return self.data[variable][t, trial]

# def set(
#     self,
#     *,
#     value,
#     variable=None,
#     t=None,
#     trial=None,
# ):
#     if (variable, t, trial) == (None, None, None):
#         raise ValueError("Attempting to replace all data")

#     if (t, trial) == (None, None):
#         # value is a numpy array
#         self.data[variable] = value
#         return

#     if (variable, trial) == (None, None):
#         # value is a dictionary of numpy arrays
#         for es, v in value.items():
#             self.data[k][t] = v
#         return

#     if (variable, t) == (None, None):
#         # value is a dictionary of numpy arrays
#         for es, v in value.items():
#             self.data[k][:, trial] = v
#         return

#     if trial == None:
#         # value is a numpy array
#         self.data[variable][t] = value
#         return

#     if t == None:
#         # value is a numpy array
#         self.data[variable][:, trial] = value
#         return

#     if variable == None:
#         # value is a dictionary
#         for k, v in value.items():
#             self.data[k][t, trial] = v
#         return

#     # value is a scalar
#     self.data[variable][t, trial] = value


# if __name__ == "__main__":

# import numpy as np

# exp_set = [("X"), ("Z"), ("X", "Z")]
# nT = 4
# A = gDict(exp_set, default=lambda: 1, x=1)
# print(A)
# print(A.get(exp_set[0]))

# B = tgDict(exp_set, nT, lambda x: np.random.randn(x), 5)
# print(B)
# print(B.get(0))
