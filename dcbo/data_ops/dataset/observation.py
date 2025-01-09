from ..base import DatasetBase
from typing import Callable, Union, Optional
import numpy as np
from sems import SEMBase
from data_struct import hDict
from collections import OrderedDict


class DatasetObs(DatasetBase):
    # TODO: remove nT and n_samples, because they should be derived from the dataX / dataY
    def __init__(
        self,
        nT: int,
        n_samples: int,
        initial_values: hDict = None,
        interv_levels: hDict = None,
        epsilon: hDict = None,
        dataY: hDict = None,
    ):
        super().__init__(nT, n_samples, initial_values, interv_levels, epsilon, dataY)

        ''' In DatasetBase,
        self.dataX = {
            'initial_values': initial_values,
            'interv_levels': interv_levels,
            'epsilon': epsilon,
        }
        self.dataY = dataY
        '''
    
    def update_new(self,  y, x = None):

        for k, vy in y.items():
            if x is not None and self.dataX is not None:
                for factors, values_x in  x[k].items():
                    # initial_values, interv_levels, epsilon
                    self.dataX[k][factors] = np.concat(self.dataX[k], values_x)
            self.dataY[k] = np.concat(self.dataY[k], vy)
