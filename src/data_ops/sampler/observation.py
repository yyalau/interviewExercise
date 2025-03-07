from ..base import DataSamplerBase
from typing import Union, Optional, List, Dict
import numpy as np
from sems import SEMBase
from data_struct import hDict, Var


class DSamplerObsBase(DataSamplerBase):
    def __init__(self, sem: SEMBase, nT: int, variables: List[Var], dtype: str = "float32"):
        '''
        Initializes the DSamplerObsBase object with the given SEM and variables.
        Parameters:
        -----------
        sem : SEMBase
            The SEM object.
        nT : int
            The number of time points.
        variables : List[Var]
            The variables.
        dtype : str
            The data type.
        '''

        super().__init__(sem, nT, variables, dtype)


    def sample(
        self,
        initial_values: Optional[hDict[str, Union[int, float]]] = None,
        interv_levels: Optional[hDict[str, Union[np.array]]] = None,
        epsilon: Optional[hDict[str, Union[np.array]]] = None,
        n_samples: int = 1,
    ) -> hDict:
        '''
        Samples the data.
        Parameters:
        -----------
        initial_values : Optional[hDict[str, Union[int, float]]]
            The initial values.
        interv_levels : Optional[hDict[str, Union[np.array]]]
            The intervention levels.
        epsilon : Optional[hDict[str, Union[np.array]]]
            The epsilon values.
        n_samples : int
            The number of samples.
        Returns:
        --------
        samples : hDict
            The samples.
        '''
        
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
            default=lambda x, y, _: np.zeros((x, y), dtype=self.dtype),
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
    
    def sc_sample(
        self,
        initial_values: Optional[hDict[str, Union[int, float]]],
        interv_levels: Optional[hDict[str, Union[np.array]]],
        epsilon: Optional[hDict[str, Union[np.array]]],
        n_samples: int
    ) -> None:
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


class DSamplerObsBF(DSamplerObsBase):
    def __init__(self, sem: SEMBase, variables: List[str], dtype: str = "float32"):
        '''
        Initializes the DSamplerObsBF object with the given SEM and variables.
        Parameters:
        -----------
        sem : SEMBase
            The SEM object.
        variables : List[str]
            The variables.
        dtype : str
            The data type.
        '''
        
        super().__init__(sem, 1, variables, dtype)

    def select_value(
        self,
        sem_func: callable,
        init: Union[int, float],
        interv: Union[int, float],
        epsilon: np.array,
        t: int,
        samples: hDict,
        n_samples: int
    ) -> np.array:
        '''
        Selects the value for the given variables.
        Parameters:
        -----------
        sem_func : callable
            The SEM function.
        init : Union[int, float]
            The initial value for SEM.
        interv : Union[int, float]
            The intervention level for SEM.
        epsilon : np.array
            The epsilon values for SEM.
        t : int
            The time point.
        samples : hDict
            The samples.
        n_samples : int
            The number of samples.
        Returns:
        --------
        value : np.array
            The value.
        '''
        return sem_func(epsilon, samples).astype(self.dtype)

    def sample(self, epsilon: hDict[str, Union[np.array]], n_samples: int) -> hDict:
        '''
        Samples the data.
        Parameters:
        -----------
        epsilon : hDict[str, Union[np.array]]
            The epsilon values.
        n_samples : int
            The number of samples.
        Returns:
        --------
        samples : hDict
            The samples.
        '''
        return super().sample(
            initial_values=None,
            interv_levels=None,
            epsilon=epsilon,
            n_samples=n_samples,
        )


class DSamplerObsDCBO(DSamplerObsBase):
    def __init__(self, sem: SEMBase, nT: int, variables: List[str], dtype: str = "float32"):
        super().__init__(sem, nT, variables, dtype)
        self.static_epsilon = hDict(
            variables=variables, nT=nT, nTrials=1, default=lambda x, y, dtype: np.zeros((x, y), dtype = dtype)
        )

    def select_value(
        self,
        sem_func: callable,
        init: Union[int, float],
        interv: Union[int, float],
        epsilon: np.array,
        t: int,
        samples: hDict,
        n_samples: int
    ) -> np.array:
        '''
        Selects the value for the given variables.
        Parameters:
        -----------
        sem_func : callable
            The SEM function.
        init : Union[int, float]
            The initial value for SEM.
        interv : Union[int, float]
            The intervention level for SEM.
        epsilon : np.array
            The epsilon values for SEM.
        t : int
            The time point.
        samples : hDict
            The samples.
        n_samples : int
            The number of samples.
        Returns:
        --------
        value : np.array
            The value.
        '''
        if init is not None and not np.isnan(init):
            return np.array([init] * n_samples)

        if interv is not None and not np.isnan(interv):
            return np.array([interv] * n_samples)

        return sem_func(epsilon, t, samples).astype(self.dtype)

    def sample(
        self,
        initial_values: Optional[hDict[str, Union[int, float]]],
        interv_levels: Optional[hDict[str, Union[np.array]]],
        n_samples: int,
        epsilon: Optional[hDict[str, Union[np.array]]] = None,
    ) -> hDict:
        '''
        Samples the data.
        Parameters:
        -----------
        epsilon : hDict[str, Union[np.array]]
            The epsilon values.
        n_samples : int
            The number of samples.
        Returns:
        --------
        samples : hDict
            The samples.
        '''
        return super().sample(
            initial_values,
            interv_levels,
            epsilon if epsilon is not None else self.static_epsilon,
            n_samples,
        )
