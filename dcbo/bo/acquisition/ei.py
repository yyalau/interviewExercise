
import numpy as np
import tensorflow_probability as tfp
gpEI = tfp.experimental.bayesopt.acquisition.GaussianProcessExpectedImprovement
from .base import EIBase

class CausalEI(EIBase):
    def __init__(
        self,
        bo_model,
        task = "min",
        jitter: float = 0.0,
    ) -> None:
        """
        This acquisition computes for a given input the improvement over the current best observed value in
        expectation. For more information see:

        Efficient Global Optimization of Expensive Black-Box Functions
        Jones, Donald R. and Schonlau, Matthias and Welch, William J.
        Journal of Global Optimization

        :param model: model that is used to compute the improvement.
        :param jitter: parameter to encourage extra exploration.
        """
        super().__init__(task, jitter)
        self.bo_model = bo_model

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Expected Improvement.

        :param x: points where the acquisition is evaluated.
        """
        gprm = self.bo_model.gprm()(x)
        gpei = gpEI(predictive_distribution = gprm, 
             observations = gprm.observations,
             exploration = self.jitter)
        
        improvement = gpei() # sames as gpei(x)
        
        return improvement.numpy()

    # def evaluate_with_gradients(self, gp_model, gp_grad, x: np.ndarray, ):
    #     """
    #     Computes the Expected Improvement and its derivative.

    #     :param x: locations where the evaluation with gradients is done.
    #     """

    #     dmean_dx, dvariance_dx = gp_grad
    #     dstandard_deviation_dx = dvariance_dx / (2 * standard_deviation)

    #     mean += self.jitter
    #     u, pdf, cdf = self.stats(self.current_global_min, mean, standard_deviation)
    #     if self.task == "min":
    #         dimprovement_dx = dstandard_deviation_dx * pdf - cdf * dmean_dx
    #     else:
    #         dimprovement_dx = -(dstandard_deviation_dx * pdf - cdf * dmean_dx)

    #     improvement = self.evaluate(gp_model, x)
    #     return improvement, dimprovement_dx

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients."""
        # return isinstance(self.model, IDifferentiable)
        raise AttributeError("should not be implicitly called")