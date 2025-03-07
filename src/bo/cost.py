import numpy as np
from typing import Callable, Dict, List, Tuple
from data_struct import Var

class CostBase:
    def __init__(self, cost_f: Dict[str, Callable[Var, float]]):
        '''
        Initializes the CostBase object with the given cost functions.
        Parameters:
        -----------
        cost_f : Dict[str, Callable[[Var], float]]
            The cost functions for each variable.
        '''
        
        self.cost_f = cost_f
    
    def evaluate(self, es: List[str], x: np.ndarray) -> float:
        '''
        Evaluates the cost function for the given variables and values.
        Parameters:
        -----------
        es : List[str]
            The variables for which to evaluate the cost function.
        x : np.ndarray
            The values for the variables.
        Returns:
        --------
        cost : float
            The cost of the given values for the variables.
        '''
        
        cost = 0
        for i, var in enumerate(es):
            cost += self.cost_f[var](x[:, i])
        return cost


class FixedCost(CostBase):
    def __init__(self, variables: List[str], c: float):
        '''
        Initializes the FixedCost object with the given variables and cost.
        Parameters:
        -----------
        variables : List[str]
            List of variables.            
        c : float
            The fixed cost.
        '''
        cost_f = {var: lambda _: c for var in variables}
        super().__init__(cost_f)

class RandomCost(CostBase):
    def __init__(self, variables: List[str], c_range: Tuple[int, int] = (1, 10)):
        '''
        Initializes the RandomCost object with the given variables and cost range.
        Parameters:
        -----------
        variables : List[str]
            List of variables.
        c_range : Tuple[int, int]
            The range of the random cost.
        '''
        cost_f = {var: lambda _: np.random.randint(*c_range) for var in variables}
        super().__init__(cost_f)
        
class InvValueCost(CostBase):
    def __init__(self, variables: List[str], c: float):
        '''
        Initializes the InvValueCost object with the given variables and cost.
        Parameters:
        -----------
        variables : List[str]
            List of variables.
        c : float
            The cost per unit value.
        '''
        cost_f = {var: lambda x: np.sum(np.abs(x)) + c for var in variables}
        super().__init__(cost_f)