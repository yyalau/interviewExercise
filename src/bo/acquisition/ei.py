
import numpy as np
import tensorflow_probability as tfp
gpEI = tfp.experimental.bayesopt.acquisition.GaussianProcessExpectedImprovement
from .base import EIBase
import tensorflow as tf
from models import BOModel

class CausalEI(EIBase):
    def __init__(
        self,
        bo_model: BOModel,
        task: str = "min",
        jitter: float = 0.0,
    ) -> None:
        """
        This acquisition computes for a given input the improvement over the current best observed value in
        expectation. For more information see:

        Efficient Global Optimization of Expensive Black-Box Functions
        Jones, Donald R. and Schonlau, Matthias and Welch, William J.
        Journal of Global Optimization
        
        Parameters:
        -----------
        bo_model : BOModel
            The Bayesian optimization model.
        task : str
            Indicates whether the task is minimization ("min") or maximization ("max").
        jitter : float
            A small value to add to the variance to avoid numerical issues.

        """
        super().__init__(task, jitter)
        self.bo_model = bo_model
        

    def evaluate(self, x: np.ndarray) -> np.array:
        '''
        Computes the Expected Improvement (EI) at the given points.
        This method evaluates the acquisition function at the specified points
        by predicting the mean and variance using the Bayesian optimization model,
        and then calculating the expected improvement.

        Parameters:
        -----------
        x : np.ndarray
            Points where the acquisition function is evaluated.
        
        Returns:
        --------
        np.array
            The expected improvement values at the given points.    
        '''
        mean, variance = self.bo_model.predict(x)        
        self.cmin = tf.reduce_min(self.bo_model.Y)
        
        return super().evaluate(mean, variance)


