import numpy as np
from .base import EIBase
import tensorflow as tf
from typing import Callable, Optional



class ManualCausalEI(EIBase):
    def __init__(
        self,
        v_target: float,
        mean_function: Callable[[np.ndarray], tf.Tensor],
        variance_function: Callable[[np.ndarray], tf.Tensor],
        previous_variance: float = 1.0,
        task: str = "min",
        cmin: Optional[float] = None,
        jitter: float = 0.0,
    ) -> None:
        """
        The improvement when a BO model has not yet been instantiated.

        Efficient Global Optimization of Expensive Black-Box Functions
        Jones, Donald R. and Schonlau, Matthias and Welch, William J.
        Journal of Global Optimization
        
        Parameters:
        -----------
        v_target: float
            The target value.
        mean_function: Callable[[np.ndarray], tf.Tensor]
            The mean function for the current DCBO exploration at given temporal index.
        variance_function: Callable[[np.ndarray], tf.Tensor]
            The mean function for the current DCBO exploration at given temporal index.
        previous_variance: float
            The previous variance.
        task: str
            Indicates whether the task is minimization ("min") or maximization ("max").
        cmin: Optional[float]
            The minimum value.
        jitter: float
            A small value to add to the variance to avoid numerical issues.
        """        

        super().__init__(task, jitter)
        
        self.v_target = v_target
        self.mean_function = mean_function
        self.variance_function = variance_function
        self.previous_variance = previous_variance
        self.cmin = cmin

    def evaluate(self, samples: np.array) -> np.ndarray:
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """
        mean = self.mean_function(samples) + self.jitter
        variance = self.variance_function(samples) + self.previous_variance         
        return super().evaluate(mean, variance)
    