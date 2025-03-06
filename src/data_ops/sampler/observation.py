from ..base import DataSamplerBase
from typing import Union, Optional, List
import numpy as np
from sems import SEMBase
from data_struct import hDict, Var


class DSamplerObsBase(DataSamplerBase):
    def __init__(self, sem: SEMBase, nT: int, variables: List[Var], dtype="float32"):

        super().__init__(sem, nT, variables, dtype)
        # assert isinstance(variables, list) and all(isinstance(v, Var) for v in variables), "variables must be a list of Variables"

    def sc_sample(self, initial_values, interv_levels, epsilon, n_samples):
        assert initial_values is None or isinstance(
            initial_values, hDict
        ), "initial_values must be None or a dictionary"
        assert not (
            initial_values is not None and interv_levels is not None
        ), "Cannot have both initial values and interv_levels"
        assert interv_levels is None or isinstance(
            interv_levels, hDict
        ), "interv_levels must be None or a dictionary"
        assert epsilon is None or isinstance(
            epsilon, hDict
        ), "epsilon must be None or a dictionary"
        assert (
            isinstance(n_samples, int) and n_samples > 0
        ), "n_samples must be a positive integer"

    def sample(
        self,
        initial_values: Optional[hDict[str, Union[int, float]]] = None,
        interv_levels: Optional[hDict[str, Union[np.array]]] = None,
        epsilon: Optional[hDict[str, Union[np.array]]] = None,
        n_samples=1,
    ):
        if epsilon is None:
            epsilon = hDict(
                variables=self.variables,
                nT=self.nT,
                nTrials=n_samples,
                default=np.random.randn,
            )

        if initial_values is None:
            initial_values = hDict(variables=self.variables)

        if interv_levels is None:
            interv_levels = hDict(variables=self.variables, nT=self.nT)

        samples = hDict(
            variables=self.variables,
            nT=self.nT,
            nTrials=n_samples,
            default=lambda x, y: np.zeros((x, y), dtype=self.dtype),
        )

        for t in range(self.nT):
            sem_func = self.sem.dynamic() if t > 0 else self.sem.static()

            for var, f_sem in sem_func.items():
                samples[var][t, :] = self.select_value(
                    f_sem,
                    initial_values[var][0, 0],
                    interv_levels[var][t, 0],
                    epsilon[var][t, :],
                    t,
                    samples,
                    n_samples,
                )

        return samples


class DSamplerObsBF(DSamplerObsBase):
    def __init__(self, sem: SEMBase, variables: list[str], dtype="float32"):
        super().__init__(sem, 1, variables, dtype)

    def select_value(
        self, sem_func, init, interv, epsilon, t, samples: hDict, n_samples
    ):
        return sem_func(epsilon, samples).astype(self.dtype)

    def sample(self, epsilon, n_samples):
        return super().sample(
            initial_values=None,
            interv_levels=None,
            epsilon=epsilon,
            n_samples=n_samples,
        )


class DSamplerObsDCBO(DSamplerObsBase):
    def __init__(self, sem: SEMBase, nT: int, variables: list[str], dtype="float32"):
        super().__init__(sem, nT, variables, dtype)
        self.static_epsilon = hDict(
            variables=variables, nT=nT, nTrials=1, default=lambda x, y: np.zeros((x, y))
        )

    def select_value(
        self, sem_func, init, interv, epsilon, t, samples: hDict, n_samples
    ):
        if init is not None and not np.isnan(init):
            return np.array([init] * n_samples)

        if interv is not None and not np.isnan(interv):
            return np.array([interv] * n_samples)

        return sem_func(epsilon, t, samples).astype(self.dtype)

    def sample(
        self,
        initial_values,
        interv_levels,
        n_samples,
        epsilon=None,
    ):
        return super().sample(
            initial_values,
            interv_levels,
            epsilon if epsilon is not None else self.static_epsilon,
            n_samples,
        )
