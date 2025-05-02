import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from .base import NLLBase
from typing import Callable, Optional, Union, Tuple


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
        
        assert callable(kernel_fn), "kernel_fn must be a callable kernel function"
        assert isinstance(feature_ndims, int) and feature_ndims > 0, "feature_ndims must be a positive integer"
        if mean_fn is not None:
            assert callable(mean_fn), "mean_fn must be callable or None"
        assert isinstance(variance, (float, int)) and variance > 0, "variance must be a positive number"
        assert isinstance(lengthscale, (float, int)) and lengthscale > 0, "lengthscale must be a positive number"
        assert isinstance(noise_var, (float, int)) and noise_var > 0, "noise_var must be a positive number"
        assert isinstance(dtype, str) and dtype in ['float32', 'float64'], "Invalid dtype. Must be 'float32' or 'float64'."

        self.minY = np.inf
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
        x: Union[np.ndarray, tf.Tensor],    
        y: Union[np.ndarray, tf.Tensor],
        n_restart: int = 10,
        verbose: bool = False,
    ) -> "GPRegression":
        """
        Fit the Gaussian Process model to the data.

        Parameters:
        ----------
            x: np.ndarray or tf.Tensor
                Input data points.
            y: np.ndarray or tf.Tensor
                Target values.
            n_restart: int, optional
                Number of restarts for optimization. Defaults to 10.
            verbose: bool, optional
                Whether to print verbose output. Defaults to False.

        Returns:
        -------
            GPRegression
                The fitted model instance.
        """
        assert isinstance(x, (np.ndarray, tf.Tensor)), "Input x must be a numpy ndarray or tf.Tensor"
        assert isinstance(y, (np.ndarray, tf.Tensor)), "Input y must be a numpy ndarray or tf.Tensor"
        assert x.shape[0] == y.shape[0], "x and y must have the same number of samples"
        assert x.ndim in [1, 2], "x must be 1D or 2D"
        assert y.ndim in [1, 2], "y must be 1D or 2D"
        assert isinstance(n_restart, int), "n_restart must be an integer"
        assert isinstance(verbose, bool), "verbose must be a boolean"
        
        x = tf.cast(x, self.dtype) 
        y = tf.cast(y, self.dtype)
                
        self.minY = min(tf.reduce_min(y), self.minY)
        # We'll use an unconditioned GP to train the kernel parameters.        
        super().fit(x, y, n_restart=n_restart, verbose=verbose)      
        
        if verbose:
            self.logging()

        return self

    def logging(self) -> None:
        """
        Log the final trained parameters.
        """
        
        print("Trained parameters:")
        print("amplitude: {}".format(self.amplitude._value().numpy()))
        print("length_scale: {}".format(self.length_scale._value().numpy()))
        print(
            "observation_noise_variance: {}".format(
                self.observation_noise_variance._value().numpy()
            )
        )

    def predict(self, x: Union[np.ndarray, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
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
        assert isinstance(x, (np.ndarray, tf.Tensor)), "Input x must be a numpy ndarray or tf.Tensor"
        assert x.ndim in [1, 2], "x must be 1D or 2D"
        assert x.shape[0] > 0, "x must have at least one sample"
                
        if self.feature_ndims == 1 and x.ndim == 1 or self.feature_ndims > 1 and x.ndim == 2:
            x = x[..., None]
        gp_fit = self.model.get_marginal_distribution(x)   
        return gp_fit.mean(), gp_fit.variance()
