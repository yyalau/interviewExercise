import numpy as np
class CostBase:
    def __init__(self, cost_f):
        self.cost_f = cost_f
    
    def evaluate(self, es, x):
        cost = 0
        for i, var in enumerate(es):
            cost += self.cost_f[var](x[:, i])
        return cost


class FixedCost(CostBase):
    def __init__(self, variables, c):
        cost_f = {var: lambda _ : c for var in variables}
        super().__init__(cost_f)

class RandomCost(CostBase):
    def __init__(self, variables, c_range = (1, 10)):
        cost_f = {var: lambda _ : np.random.randint(*c_range) for var in variables}
        super().__init__(cost_f)
        
class InvValueCost(CostBase):
    def __init__(self, variables, c):
        cost_f = {var: lambda x : np.sum(np.abs(x)) + c for var in variables}
        super().__init__(cost_f)
        