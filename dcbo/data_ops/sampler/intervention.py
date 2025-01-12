from ..base import DataSamplerBase
from typing import Callable, Union, Optional
import numpy as np
from sems import SEMBase
from data_struct import hDict, Node, Var
from collections import OrderedDict


class DSamplerInv:
    def __init__(self, semhat):
        # super().__init__(semhat, nT, n_samples, variables)
        self.sem = semhat
        self.nT = semhat.nT
        self.variables = semhat.vs
        # self.node_parents = semhat.get_edgekeys

    # TODO: wrong implementation
    def sample(
        self,
        initial_values: Optional[dict[str, Union[int, float]]] = None,
        interv_levels: Optional[dict[str, Union[np.array]]] = None,
        n_samples=1,
        moment=0,
    ):
        
        samples = hDict(
            variables=self.variables,
            nT=self.nT,
            nTrials=n_samples,
            default=lambda x, y: np.zeros((x, y)),
        )
        
        if initial_values is None:
            initial_values = hDict(variables=self.variables)

        if interv_levels is None:
            interv_levels = hDict(variables=self.variables, nT=self.nT)

        for t in range(self.nT):
            # import ipdb; ipdb.set_trace()
            sem_func = self.sem.dynamic(moment) if t >0 else self.sem.static(moment)
            for var, function in sem_func.items():
                samples[var][t, :] = self.select_value(
                    function[0,0],
                    initial_values[var][0, 0],
                    interv_levels[var][t, 0],
                    var,
                    t,
                    samples,
                    n_samples,
                ).reshape(-1)        
        return samples
        
    
    def select_value(self, function, init, interv, var, t, samples: hDict, n_samples):
        if init is not None:
            return np.array([init] * n_samples)

        if interv is not None:
            return np.array([interv] * n_samples)
        
        node = Node(var, t)
                
        edge_key_t = self.sem.get_edgekeys(node, t)
        edge_key_t1 = self.sem.get_edgekeys(node, t-1)
        
        # emission only
        if t ==0 and edge_key_t:
            return function(t, None, edge_key_t, samples, n_samples)
        
        # source only
        if not edge_key_t1 and not edge_key_t:
            return function(t, (None, var), n_samples)
        
        # transition only
        return function(t, edge_key_t1, edge_key_t, samples, n_samples)
