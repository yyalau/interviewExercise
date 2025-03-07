import numpy as np  
from typing import Tuple
import tensorflow_probability as tfp
tfd = tfp.distributions
import tensorflow as tf 
# import tfp.experimental.bayesopt.acquisition as tfpbo

class EIBase:
    """
    A class to represent the Expected Improvement (EI) base for Bayesian Optimization.

    Attributes:
    -----------
    task : int
        Indicates whether the task is minimization (1) or maximization (-1).
    jitter : float
        A small value to add to the variance to avoid numerical issues.

    Methods:
    --------
    __init__(task: int, jitter: float) -> None:
        Initializes the EIBase object with the given task and jitter.
    get_stats(x: int, mean: tf.Tensor, sd: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        Returns the pdf and cdf of the standard normal distribution evaluated at (x - mean)/sigma.
    clipv(variance: tf.Tensor) -> tf.Tensor:
        Clips the variance to avoid negative or NaN values.
    evaluate(mean: tf.Tensor, variance: tf.Tensor) -> tf.Tensor:
        Evaluates the expected improvement given the mean and variance.
    """
    def __init__(self, task: int, jitter: float) -> None:
        '''
        Initializes the EIBase object with the given task and jitter.
        Parameters:
        -----------
        task : int
            Indicates whether the task is minimization (1) or maximization (-1).
        jitter : float
            A small value to add to the variance to avoid numerical issues.
        '''
        self.task = 1 if task == "min" else -1
        self.jitter = jitter
        
    def get_stats(
        self,
        x: int, mean: tf.Tensor, sd: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        
        '''
        Returns the pdf and cdf of the standard normal distribution evaluated at (x - mean)/sigma.
        Parameters:
        -----------
        x : int
            The value at which to evaluate the distribution.
        mean : tf.Tensor
            The mean of the distribution.
        sd : tf.Tensor
            The standard deviation of the distribution.
        
        Returns:
        --------
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
            The z-score, pdf, and cdf of the standard normal distribution.
        '''

        dist = tfd.Normal(loc=mean, scale=sd)
        
        z = (x - mean) / sd
        pdf = dist.prob(z)
        cdf = dist.cdf(z)
        return z, pdf, cdf

    def clipv(self, variance: tf.Tensor) -> tf.Tensor:
        '''
        Clips the variance to avoid negative or NaN values.
        Parameters:
        -----------
        variance : tf.Tensor
            The variance to clip.
        
        Returns:
        --------
        tf.Tensor
            The clipped variance.
        '''
        
        if tf.reduce_any(nan_v := tf.math.is_nan(variance)):
            variance = tf.where(nan_v, tf.zeros_like(variance), variance)
        elif tf.reduce_any(variance < 0):
            variance = variance.clip(0.0001)
        return variance

    def evaluate(self, mean: tf.Tensor, variance: tf.Tensor) -> np.array:
        '''
        Evaluates the expected improvement given the mean and variance.
        Parameters:
        -----------
        mean : tf.Tensor
            The mean of the distribution.
        variance : tf.Tensor
            The variance of the distribution.
            
        Returns:
        --------
        np.array
            The expected improvement.
            
        '''
        sd = tf.math.sqrt(self.clipv(variance))
        u, pdf, cdf = self.get_stats(self.cmin if self.cmin is not None else mean, mean, sd) 
        improvement = self.task * sd * (u * cdf + pdf)
        return tf.reshape(improvement, -1).numpy()
