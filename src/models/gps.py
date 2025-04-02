import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from .base import NLLBase
from typing import Callable, Optional, Union


tfb = tfp.bijectors
tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels


class GPRegression(NLLBase):
    def __init__(
        self,
        kernel_fn: Callable,
        feature_ndims: int,
        mean_fn: Optional[Callable] = None,
        variance: float = 1.0,
        lengthscale: float = 1.0,
        noise_var: float = 1.0,
        dtype: str = "float32",
    ):
        """
        Initialize the GPRegression model.

        Parameters:
        ----------
            kernel_fn: Callable
                Kernel function for the Gaussian Process.
            feature_ndims: int
                Number of feature dimensions.
            mean_fn: Optional[Callable]
                Mean function for the Gaussian Process. Defaults to None.
            variance: float
                Initial variance for the kernel. Defaults to 1.0.
            lengthscale: float
                Initial lengthscale for the kernel. Defaults to 1.0.
            noise_var: float
                Initial observation noise variance. Defaults to 1.0.
            dtype: str
                Data type for the model. Defaults to "float32".
        """
        self.X = None
        self.Y = None
        self.mean_fn = mean_fn
        self.dtype = dtype

        self.amplitude = tfp.util.TransformedVariable(
            variance, tfb.Exp(), dtype=dtype, name="amplitude"
        )
        
        self.length_scale = tfp.util.TransformedVariable(
            lengthscale, tfb.Exp(), dtype=dtype, name="length_scale"
        )
        
        self.observation_noise_variance = tfp.util.TransformedVariable(
            noise_var, tfb.Exp(), dtype=dtype, name="observation_noise_variance"
        )

        self.kernel = kernel_fn(
            amplitude=self.amplitude,
            length_scale=self.length_scale,
            feature_ndims=feature_ndims,
        )
        
        model = tfd.GaussianProcess(
            kernel=self.kernel,
            mean_fn=self.mean_fn,
            index_points=None,
            observation_noise_variance=self.observation_noise_variance,
        )
        
        super().__init__(model=model, feature_ndims=feature_ndims, dtype=dtype)
        
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_restart: int = 10,
        verbose: bool = False,
    ) -> float:
        """
        Fit the Gaussian Process model to the data.

        Parameters:
        ----------
            x: np.ndarray
                Input data points.
            y: np.ndarray
                Target values.
            n_restart: Optional[int]
                Number of restarts for optimization. Defaults to 10.
            verbose: Optional[bool]
                Whether to print verbose output. Defaults to False.

        Returns:
        -------
            float
                Negative log-likelihood of the fitted model.
        """
        self.X = x = tf.cast(x, self.dtype) 
        self.Y = y = tf.cast(y, self.dtype)
        
        # We'll use an unconditioned GP to train the kernel parameters.        
        nll = super().fit(x, y, n_restart=n_restart, verbose=verbose)      
        
        if verbose:
            self.logging(nll)

        return nll

    def logging(self, nll: float) -> None:
        """
        Log the final negative log-likelihood and trained parameters.

        Parameters:
        ----------
            nll: float
                Final negative log-likelihood value.
        """
        print("Final NLL = {}".format(nll))
        print("Trained parameters:")
        print("amplitude: {}".format(self.amplitude._value().numpy()))
        print("length_scale: {}".format(self.length_scale._value().numpy()))
        print(
            "observation_noise_variance: {}".format(
                self.observation_noise_variance._value().numpy()
            )
        )

    def predict(self, x: np.ndarray) -> tuple:
        """
        Predict the mean and variance of the Gaussian Process at new points.

        Parameters:
        ----------
            x: np.ndarray or tf.Tensor
                Input data points for prediction.

        Returns:
        -------
            tuple
                Mean and variance of the predictions.
        """
        if self.feature_ndims == 1 and x.ndim == 1 or self.feature_ndims > 1 and x.ndim == 2:
            x = x[..., None]
        gp_fit = self.model.get_marginal_distribution(x)   
        return gp_fit.mean(), gp_fit.variance()
