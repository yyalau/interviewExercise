from typing import  Union, Optional
import numpy as np
from sems import SEMBase
from data_struct import hDict, Node, Var
from data_ops import DSamplerInv
from copy import deepcopy

class Surrogate:
    def __init__(self, semhat):
        self.sem = semhat
        self.nT = semhat.nT
        self.variables= semhat.vs
        self.data_sampler = DSamplerInv(semhat)
        
    def create(self, t, interv_levels, es):
        # objective: get mean_f[t, null]
        # isampled: [t, N]
                
        
        def __sample(ilvl, interv_levels, moment = 0):
            
            new_ilvls = deepcopy(interv_levels)
            
            n_samples = ilvl.shape[0]
            if new_ilvls is None:
                new_ilvls = hDict(variables=self.variables, nT=self.nT, nTrials=n_samples)
            else:
                if (new_ilvls.nTrials  != n_samples):   
                    new_ilvls.duplicate(nTrials=n_samples)
            
            if es is not None:
                for vid, var in enumerate(es):
                    new_ilvls[var][t,:] = ilvl[:, vid]
            
            samples = hDict(
                variables=self.variables,
                nT=self.nT,
                nTrials=n_samples,
                default=lambda x, y: np.zeros((x, y)),
            )
            
            # original repo does not use initial_values            
            # if initial_values is None:
            #     initial_values = hDict(variables=self.variables)


            # import ipdb; ipdb.set_trace()
            sem_func = self.sem.dynamic(moment) if t >0 else self.sem.static(moment)
            for var, function in sem_func.items():
                samples[var][t,:] = self.select_value(
                    function[0,0],
                    new_ilvls[var][t, :],
                    var,
                    t,
                    samples,
                    n_samples,
                )
            
            samples.reduce(t=t)
            # for var in self.variables:
            #     samples[var] = np.array(samples[var][t,:] ) 
            del new_ilvls
            return samples
        
        
        
        mean = lambda ilvls: __sample(ilvls, interv_levels, 0)
        variance = lambda ilvls: __sample(ilvls, interv_levels, 1)
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