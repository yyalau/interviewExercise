from typing import Union, Optional, Tuple, Callable, Optional
import numpy as np
from .semhat import SEMHat
from data_struct import hDict, Node, Var

from copy import deepcopy
import tensorflow as tf
from tensorflow.python.framework.ops import SymbolicTensor
from utils.tools import eager_replace


class Surrogate:
    def __init__(self, semhat: SEMHat):
        """
        Parameters:
        -----------
        semhat: SEMHat
            The SEMHat object.
        dtype: str
            The data type of the surrogate model.
        """
        assert isinstance(semhat, SEMHat), f"Expected SEMHat, got {type(semhat)}"

        self.sem = semhat
        self.nT = semhat.nT
        self.variables = semhat.vs
        self.dtype = semhat.dtype

    def create(
        self,
        t: int,
        interv_levels: hDict[Var, Union[np.ndarray, tf.Tensor]],
        es: Tuple[Var, ...],
        target_var: Var,
    ) -> Tuple[
        Callable[[Union[np.ndarray, tf.Tensor]], tf.Tensor],
        Callable[[Union[np.ndarray, tf.Tensor]], tf.Tensor],
    ]:
        """
        Create mean and variance functions for the surrogate model.

        Parameters
        ----------
        t : int
            Time step.
        interv_levels : hDict[Var, Union[np.ndarray, tf.Tensor]] or None
            Intervention levels.
        es : Tuple[Var, ...]
            Tuple of variables for intervention.
        target_var : Var
            Target variable.

        Returns
        -------
        tuple
            Mean and variance functions.
        """

        assert isinstance(t, int), "Time step must be an integer."
        assert t >= 0 and t < self.nT, "Time step must be between 0 and nT-1."
        assert isinstance(es, tuple), "Intervention variables must be a tuple."

        if interv_levels is not None:
            assert isinstance(
                interv_levels, hDict
            ), "Intervention levels must be a numpy array or a tensor."
        assert len(es) > 0, "Exploration set must be provided."
        for var in es:
            assert isinstance(var, Var), "Intervention variables must be Var objects."
            assert (
                interv_levels.get(var) is not None
            ), "Intervention levels must be provided for all intervention variables."
            assert isinstance(
                interv_levels[var], (np.ndarray, tf.Tensor)
            ), "Intervention levels must be a numpy array or a tensor."
            assert (
                interv_levels[var].shape[0] == self.nT
            ), "Intervention levels must have shape (nT, ...)."
        assert isinstance(target_var, Var), "Target variable must be a Var object."
        assert target_var not in es, "Target variable cannot be in the exploration set."
        assert target_var in self.variables, "Target variable must be in the variables."
        
        assert set(es).issubset(self.variables), "Exploration set must be a subset of the variables."
        
        
        def __sample(
            ilvl: Union[np.ndarray, tf.Tensor],
            interv_levels: hDict[Var, Union[np.ndarray, tf.Tensor]],
            moment: int = 0,
        ) -> hDict:
            """
            Sample from the surrogate model.

            Parameters
            ----------
            ilvl : np.array or tf.Tensor
                Intervention levels.
            interv_levels : hDict[Var, Union[np.ndarray, tf.Tensor]] or None
                Intervention levels for each variable and time step.
            moment : int
                Moment (0 for mean, 1 for variance).

            Returns
            -------
            hDict
                Sampled values.
            """
            assert isinstance(
                ilvl, (np.ndarray, tf.Tensor)
            ), "Intervention levels must be a numpy array."
            assert ilvl.shape[1] == len(
                es
            ), "Intervention levels must have shape (nTrials, nVars)."
            assert moment in [0, 1], "Moment must be 0 for mean or 1 for variance."

            new_ilvls = deepcopy(interv_levels)
            n_samples = ilvl.shape[0]

            # Initialize new intervention levels if not provided
            if new_ilvls is None:
                new_ilvls = hDict(
                    variables=self.variables,
                    nT=self.nT,
                    nTrials=n_samples,
                    default=lambda nT, nTrials: tf.convert_to_tensor(
                        [[np.nan] * nTrials] * nT,
                        dtype=self.dtype,
                    ),
                )
            else:
                # Duplicate intervention levels if the number of trials does not match
                if new_ilvls.nTrials != n_samples:
                    new_ilvls.duplicate(nTrials=n_samples)

            # Replace intervention levels with provided values
            if es is not None:
                for vid, var in enumerate(es):
                    new_ilvls[var] = eager_replace(
                        new_ilvls[var], ilvl[:, vid], t, axis=0, dtype=self.dtype
                    )

            # Initialize samples
            samples = hDict(
                variables=self.variables,
                nT=self.nT,
                nTrials=n_samples,
                default=lambda x, y: tf.convert_to_tensor(
                    [[0.0] * y] * x, dtype=self.dtype
                ),
            )

            # Generate samples for each time step
            for hist_t in range(t + 1):
                sem_func = (
                    self.sem.dynamic(moment, hist_t) #if hist_t > 0 else self.sem.static(moment)
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

                    samples[var] = eager_replace(
                        samples[var], haha, hist_t, axis=0, dtype=self.dtype
                    )

            # Extract new samples for the target time step
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

    def select_value(
        self,
        function: Callable,
        interv: Union[np.ndarray, tf.Tensor],
        var: Var,
        t: int,
        samples: hDict,
        n_samples: int,
    ) -> tf.Tensor:
        """
        Get the value for the node, using SEMHat functions.

        Parameters
        ----------
        function : callable
            Function to compute the value.
        interv : np.ndarray or tf.Tensor
            Intervention levels.
        var : Var
            Variable name.
        t : int
            Time step.
        samples : hDict
            Sampled values.
        n_samples : int
            Number of samples.

        Returns
        -------
        tf.Tensor
            The resulting value of the node.
        """
        
        assert callable(function), "Function must be a callable."
        assert isinstance(
            interv, (np.ndarray, tf.Tensor)
        ), "Intervention levels must be a numpy array or a tensor."
        assert isinstance(var, Var), "Variable must be a Var object."
        assert isinstance(t, int), "Time step must be an integer."
        assert t >= 0 and t < self.nT, "Time step must be between 0 and nT-1."
        assert isinstance(samples, hDict), "Samples must be a hDict object."
        assert isinstance(n_samples, int), "Number of samples must be an integer."
        assert interv.shape[0] == 1 or interv.shape[0] == n_samples, (
            "Intervention levels must have shape (nTrials, ...) or (1, ...)."
        )
        assert n_samples > 0, "Number of samples must be greater than 0."
        assert interv.shape[0] == n_samples, "Intervention levels must have shape (nTrials, ...)."
        
        
        # If all intervention values are provided and not NaN, return them
        if all(v is not None and not np.isnan(v) for v in interv):
            if isinstance(interv, np.ndarray):
                return tf.convert_to_tensor(interv, dtype=self.dtype)
            return interv

        node = Node(var, t)

        edge_key_t = self.sem.get_edgekeys(node, t)
        edge_key_t1 = self.sem.get_edgekeys(node, t - 1)
        
        # Source only
        if not edge_key_t1 and not edge_key_t:
            return function((None, node), n_samples)

        # Emission or / and Transition 
        return function(edge_key_t1, edge_key_t, samples)
