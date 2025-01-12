from ..base import DataSamplerBase
from typing import Union, Optional
import numpy as np
from sems import SEMBase
from data_struct import hDict


class DSamplerObs(DataSamplerBase):
    def __init__(self, sem: SEMBase, nT: int, variables: list[str]):
        super().__init__(sem, nT,  variables)

    def sample(
        self,
        initial_values: Optional[dict[str, Union[int, float]]] = None,
        interv_levels: Optional[dict[str, Union[np.array]]] = None,
        epsilon: Optional[dict[str, Union[np.array]]] = None,
        n_samples=1,
    ):
        # assert not (initial_values is not None and interv_levels is not None), "Cannot have both initial values and interv_levels"

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
            default=lambda x, y: np.zeros((x, y)),
        )
        

        for t in range(self.nT):
            sem_func = self.sem.dynamic() if t >0 else self.sem.static()
            
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

    def select_value(self, sem_func, init, interv, epsilon, t, samples: hDict, n_samples):
        if init is not None:
            return np.array([init] * n_samples)

        if interv is not None:
            return np.array([interv] * n_samples)


        return sem_func(epsilon, t, samples)

