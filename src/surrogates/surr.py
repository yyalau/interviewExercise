from typing import Union, Optional
import numpy as np
from sems import SEMBase
from data_struct import hDict, Node, Var
# from data_ops import DSamplerInv
from copy import deepcopy
import tensorflow as tf
from tensorflow.python.framework.ops import SymbolicTensor
from utils.tools import eager_replace


class Surrogate:
    def __init__(self, semhat, dtype="float32"):
        self.sem = semhat
        self.nT = semhat.nT
        self.variables = semhat.vs
        self.dtype = dtype

    def create(self, t, interv_levels, es, target_var):
        # objective: get mean_f[t, null]
        # isampled: [t, N]

        def __sample(ilvl, interv_levels, moment=0):

            new_ilvls = deepcopy(interv_levels)
            

            n_samples = ilvl.shape[0]
            if new_ilvls is None:
                new_ilvls = hDict(
                    variables=self.variables,
                    nT=self.nT,
                    nTrials=n_samples,
                    default=lambda nT, nTrials: tf.convert_to_tensor(
                        [[np.nan] * nTrials] * nT, dtype=self.dtype,
                    ),
                )
            else:
                if new_ilvls.nTrials != n_samples:
                    new_ilvls.duplicate(nTrials=n_samples)

            if es is not None:
                for vid, var in enumerate(es):
                    # new_ilvls[var] = eager_replace(new_ilvls[var], ilvl[:, vid], t, axis=0, dtype=self.dtype)
                    new_ilvls[var] = tf.stack(
                        [
                            tf.convert_to_tensor(new_ilvls[var][tempt], dtype = self.dtype) if tempt != t else ilvl[:, vid]
                            for tempt in range(self.nT)
                        ],
                        axis=0,
                    )
                        

            samples = hDict(
                variables=self.variables,
                nT=self.nT,
                nTrials=n_samples,
                default=lambda x, y: tf.convert_to_tensor([[0.] * y] * x, dtype = self.dtype),
            )


            for hist_t in range(t + 1):
                sem_func = (
                    self.sem.dynamic(moment) if hist_t > 0 else self.sem.static(moment)
                )
                for var, function in sem_func.items():
                    haha = self.select_value(
                        function[0, 0],
                        new_ilvls[var][hist_t],
                        var,
                        hist_t,
                        samples,
                        n_samples,
                    )

                    samples[var] = eager_replace(samples[var], haha, hist_t, axis=0, dtype=self.dtype)

            new_samples = hDict(
                variables=self.variables,
                nT=1,
                nTrials=n_samples,
                default=lambda x, y: None,
            )

            for var in self.variables:
                new_samples[var] = samples[var][t]

            
            del new_ilvls
            return new_samples

        mean = lambda ilvls: __sample(ilvls, interv_levels, 0)[target_var]
        variance = lambda ilvls: __sample(ilvls, interv_levels, 1)[target_var]
        return mean, variance

    def select_value(self, function, interv, var, t, samples: hDict, n_samples):

        # n_samples = samples[var][t,:].shape[0]
        if isinstance(interv, SymbolicTensor):
            return interv
            # import ipdb; ipdb.set_trace()

        if all(v is not None and not np.isnan(v) for v in interv):

            if isinstance(interv, np.ndarray):
                return tf.convert_to_tensor(interv, dtype=self.dtype)

            return interv

        node = Node(var, t)

        edge_key_t = self.sem.get_edgekeys(node, t)
        edge_key_t1 = self.sem.get_edgekeys(node, t - 1)
        
        # emission only
        if t == 0 and edge_key_t:
            return function(t, None, edge_key_t, samples, n_samples)

        # source only
        if not edge_key_t1 and not edge_key_t:
            return function(t, (None, var), n_samples)

        # transition only
        return function(t, edge_key_t1, edge_key_t, samples, n_samples)
