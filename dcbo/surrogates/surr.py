from typing import  Union, Optional
import numpy as np
from sems import SEMBase
from data_struct import hDict, Node, Var
from data_ops import DSamplerInv

class Surrogate:
    def __init__(self, semhat):
        self.sem = semhat
        self.nT = semhat.nT
        self.variables= semhat.vs
        self.data_sampler = DSamplerInv(semhat)
        
    def create(self, t):
        # objective: get mean_f[t, null]
        # isampled: [t, N]
        
         
        def __sample(n_samples, interv_levels, moment = 0):
            
            if interv_levels is None:
                interv_levels = hDict(variables=self.variables, nT=1)
            
            samples = hDict(
                variables=self.variables,
                nT=self.nT,
                nTrials=n_samples,
                default=lambda x, y: np.zeros((x, y)),
            )
            
            # original repo does not use initial_values            
            # if initial_values is None:
            #     initial_values = hDict(variables=self.variables)

            if interv_levels is None:
                interv_levels = hDict(variables=self.variables, nT=self.nT)


            # import ipdb; ipdb.set_trace()
            sem_func = self.sem.dynamic(moment) if t >0 else self.sem.static(moment)
            for var, function in sem_func.items():
                samples[var][t,:] = self.select_value(
                    function[0,0],
                    interv_levels[var][t, :],
                    var,
                    t,
                    samples,
                    n_samples,
                )
            return samples
        
        # def __gen(moment):
        #     fs = hDict(
        #         variables=self.variables,
        #         nT=self.nT,
        #         nTrials=1,
        #     )
            
        #     # TODO: can be optimized
        #     for t in range(self.nT):
        #         sem_func = self.sem.dynamic(moment) if t >0 else self.sem.static(moment)
        #         for var, function in sem_func.items():
        #             fs[var][t, 0] = function[0,0]
            
        #     return fs        
        
        # mean = lambda n_samples: __sample(n_samples, 0)
        # variance = lambda n_samples: __sample(n_samples, 1)
        mean = lambda n_samples, interv_levels: __sample(n_samples, interv_levels, 0)
        variance = lambda n_samples, interv_levels: __sample(n_samples, interv_levels, 1)
        return mean, variance
    
    def select_value(self, function, interv, var, t, samples: hDict, n_samples):
        
        # n_samples = samples[var][t,:].shape[0]

        if all(v is not None for v in interv):
            return interv
                    
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