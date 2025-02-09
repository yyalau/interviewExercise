import numpy as np  
from typing import Tuple
import tensorflow_probability as tfp
tfd = tfp.distributions
import tensorflow as tf 
# import tfp.experimental.bayesopt.acquisition as tfpbo

class EIBase:
    def __init__(self, task, jitter):
        self.task = 1 if task == "min" else -1
        self.jitter = jitter
        
    def get_stats(
        self,
        x: np.array, mean: np.array, sd: np.array
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Returns pdf and cdf of standard normal evaluated at (x - mean)/sigma

        :param x: Non-standardized input
        :param mean: Mean to normalize x with
        :param sd: Standard deviation to normalize x with
        :return: (normalized version of x, pdf of standard normal, cdf of standard normal)
        """
        dist = tfd.Normal( loc=mean, scale=sd)
        
        z = (x - mean) / sd
        pdf = dist.prob(z)
        cdf = dist.cdf(z)
        return z, pdf, cdf

    def clipv(self, variance):
        if tf.reduce_any( nan_v := tf.math.is_nan(variance)):
            variance = tf.where(nan_v, tf.zeros_like(variance), variance)
        elif tf.reduce_any(variance < 0):
            variance = variance.clip(0.0001)
        return variance

    
    def evaluate(self, mean, variance):
        sd = np.sqrt(self.clipv(variance))
        u, pdf, cdf = self.get_stats(self.cmin, mean, sd)
        improvement = self.task * sd * (u * cdf + pdf)

        return tf.reshape(improvement, -1).numpy()
