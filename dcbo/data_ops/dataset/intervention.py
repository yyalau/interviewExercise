from .base import DataSamplerBase


class DataIntervention(DataSamplerBase):
    
    def __init__(self, n_samples, sem, nT, i_grid):
        super().__init__(n_samples, sem, nT)
        self.intervention = intervention
        
    def sample(self):
        pass