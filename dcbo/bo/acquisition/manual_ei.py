import numpy as np
from .base import EIBase

# from emukit.core.acquisition import Acquisition
# from emukit.core.interfaces import IDifferentiable, IModel


class ManualCausalEI(EIBase):
    def __init__(
        self,
        v_target,
        mean_function,
        variance_function,
        previous_variance = 1,
        task = "min",
        jitter: float = 0.0,
    ) -> None:
        """
        The improvement when a BO model has not yet been instantiated.

        Efficient Global Optimization of Expensive Black-Box Functions
        Jones, Donald R. and Schonlau, Matthias and Welch, William J.
        Journal of Global Optimization

        :param mean_function: the mean function for the current DCBO exploration at given temporal index
        :param variance_function: the mean function for the current DCBO exploration at given temporal index
        :param jitter: parameter to encourage extra exploration.
        """
        super().__init__(task, jitter)
        
        self.v_target = v_target
        self.mean_function = mean_function
        self.variance_function = variance_function
        self.previous_variance = previous_variance

    def evaluate(self, samples, cmin) -> np.ndarray:
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """
        mean = self.mean_function(samples)[self.v_target] + self.jitter
        variance = self.variance_function(samples)[self.v_target] + self.previous_variance         
        sd = np.sqrt(self.clipv(variance))

        u, pdf, cdf = self.get_stats(cmin, mean, sd)
        improvement = self.task * sd * (u * cdf + pdf)

        return improvement[0]

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition does not have gradients."""
        return False
