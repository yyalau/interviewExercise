from data_struct import hDict
import numpy as np

class DataBase:
    def __init__(self, sem, nT, variables, dtype="float32"):
        assert nT > 0, "the number of time steps (nT) must be greater than 0"

        self.sem = sem
        self.nT = nT
        self.variables = variables
        self.dtype = dtype


class DatasetBase(DataBase):

    def __init__(self, nT, n_samples, dataY, dtype="float32"):
        super().__init__(None, nT, np.array(dataY.keys()), dtype)
        self.samples = n_samples
        assert n_samples > 0, "the number of samples (n_samples) must be greater than 0"

        # TODO: become a sampling scheme
        # if dataX is None:
        #     self.dataX = hDict(
        #         variables=sem.variables,
        #         nT=nT,
        #         nTrials=n_samples,
        #     )

        # if dataY is None:
        #     self.dataY = hDict(
        #         variables=sem.variables,
        #         nT=nT,
        #         nTrials=n_samples,
        #     )

    def update_new(x, y):
        raise NotImplementedError("This method must be implemented in a subclass")

class DataSamplerBase(DataBase):

    def __init__(self, sem, nT, variables, dtype="float32"):
        super().__init__(sem, nT, variables, dtype)

    def sample(self):
        raise NotImplementedError("This method must be implemented in a subclass")
    
