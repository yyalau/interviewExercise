from .base import SEMBase
from collections import OrderedDict
import numpy as np

class X2Y(SEMBase):
    def __init__(self, A = 2, B = 1):
        super().__init__()
        self.A = A
        self.B = B
            
    def static(self):
        X = lambda noise, sample: noise
        Y = lambda noise, sample: self.A * np.tanh(self.B * sample["X"]) + noise
        return OrderedDict([("X", X), ("Y", Y)])

class Y2X(SEMBase):
    def __init__(self, A = 2, B = 1):
        super().__init__()
        self.A = A
        self.B = B

    def static(self):
        Y = lambda noise, sample: noise
        X = lambda noise, sample: self.A * np.tanh(self.B * sample["Y"]) + noise
        return OrderedDict([("Y", Y), ("X", X)])
    
class Confounder(SEMBase):
    def __init__(self, A = 2, B = 1, C = 2, D = 1):
        super().__init__()
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def static(self):
        X = lambda noise, sample: self.A * np.tanh(self.B * sample["U"]) + noise
        U = lambda noise, sample: noise
        Y = lambda noise, sample: self.C * np.tanh(self.D * sample["U"]) + noise
        return OrderedDict([ ("U", U),("X", X), ("Y", Y)])
    
