
from ..base import DatasetBase
import numpy as np
from data_struct import hDict


class DatasetObsDCBO(DatasetBase):
    def __init__(
        self,
        initial_values: hDict[np.array],
        interv_levels: hDict[np.array],
        epsilon: hDict[np.array],
        dataY: hDict[np.array],
        dtype: str="float32",
    ) -> None:
        '''
        Initializes the DatasetObsDCBO object with the given data.
        Parameters:
        -----------
        initial_values : hDict[np.array]
            The initial values.
        interv_levels : hDict[np.array]
            The intervention levels.
        epsilon : hDict[np.array]
            The epsilon values.
        dataY : hDict[np.array]
            The data.
        dtype : str
            The data type.
        '''
        self.sc(initial_values, interv_levels, epsilon, dataY)

        self.dataX = {
            "initial_values": initial_values,
            "interv_levels": interv_levels,
            "epsilon": epsilon,
        }
        self.dataY = dataY

        super().__init__(dataY.nT, dataY.nTrials, list(dataY.keys()), dtype)

    def update(self, eps: hDict, y: hDict) -> None:     
        """
        Updates the dataset with the given data.
        Parameters:
        -----------
        eps : hDict
            The epsilon values.
        y : hDict
            The data.
        """
        self.sc( self.dataX["initial_values"], self.dataX["interv_levels"], eps, y)

        f = lambda x, y: np.concatenate((x, y), axis=1)
        for var in self.variables:
            self.dataX["epsilon"][var] = f(self.dataX["epsilon"][var], eps[var])
            self.dataY[var] = f(self.dataY[var], y[var])

        self.n_samples += y[var].shape[1]

    def sc(self, initial_values: hDict, interv_levels: hDict, epsilon: hDict, dataY: hDict) -> None:
        assert isinstance(
            initial_values, hDict
        ), "initial_values must be an instance of hDict"
        assert isinstance(
            interv_levels, hDict
        ), "interv_levels must be an instance of hDict"
        assert isinstance(epsilon, hDict), "epsilon must be an instance of hDict"
        assert isinstance(dataY, hDict), "dataY must be an instance of hDict"
        get_vars = lambda x: set(x.keys())

        assert (
            get_vars(initial_values)
            == get_vars(interv_levels)
            == get_vars(epsilon)
            == get_vars(dataY)
        ), "variables must be the same for all inputs"

        assert (
            initial_values.nT == 1
        ), "initial_values only initialize the values at t=0"
        assert (
            interv_levels.nT == epsilon.nT == dataY.nT
        ), f"Timesteps must be the same for all inputs. Got {interv_levels.nT}, {epsilon.nT}, {dataY.nT}"

        assert (
            initial_values.nTrials == interv_levels.nTrials == 1
        ), f"Number of data at each time step must be 1 for initial_values and interv_levels. Got {initial_values.nTrials}, {interv_levels.nTrials}"

        assert (
            epsilon.nTrials == dataY.nTrials
        ), f"Number of data points at each timestep must be the same for all inputs. Got {initial_values.nTrials}, {interv_levels.nTrials}, {epsilon.nTrials}, {dataY.nTrials}"

