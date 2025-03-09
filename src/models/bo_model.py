from . import GPRegression, GaussianRBF, GammaRBF
import tensorflow_probability as tfp
import tensorflow as tf
from data_struct import Var
from typing import Callable, Tuple, Union

tfd = tfp.distributions


class BOModel(GPRegression):
    def __init__(
        self,
        mean_f: Callable[[float], float],
        variance_f: Callable[[float], float],
        variance: float = 1.0,
        lengthscale: float = 0.5,
        noise_var: float = 1.0,
        alpha: float = 2,
        use_gamma_prior: bool = True,
        dtype: str = "float32",
    ):  
        """
        Bayesian Optimization model inheriting from GPRegression.
        
        Parameters:
        mean_f (Callable[[float], float]): 
            Function representing the mean of the model.
        variance_f (Callable[[float], float]): 
            Function representing the variance of the model.
        variance (float): 
            Initial variance of the kernel.
        lengthscale (float): 
            Initial lengthscale of the kernel.
        noise_var (float): 
            Initial noise variance of the kernel.
        alpha (float): 
            Power parameter of the kernel function.
        use_gamma_prior (bool): 
            Flag to determine if Gamma prior is used for the kernel.
        dtype (str): 
            Data type of the model, either 'float32' or 'float64'.
        """
        
        assert isinstance(mean_f, Callable), "mean_f must be a callable"
        assert isinstance(variance_f, Callable), "variance_f must be a callable"
        assert isinstance(variance, (int, float)), "variance must be a number"
        assert isinstance(lengthscale, (int, float)), "lengthscale must be a number"
        assert isinstance(noise_var, (int, float)), "noise_var must be a number"
        assert isinstance(alpha, (int, float)), "alpha must be a number"
        assert isinstance(use_gamma_prior, bool), "use_gamma_prior must be a boolean"
        assert isinstance(dtype, str), "dtype must be a string"
        assert dtype in ["float32", "float64"], "dtype must be 'float32' or 'float64"
        
        def kernel_fn(
            amplitude: tf.Variable,
            length_scale: tf.Variable,
            feature_ndims: tf.Variable,
        ) -> tfp.math.psd_kernels.AutoCompositeTensorPsdKernel:
            ClassRBF = GammaRBF if use_gamma_prior else GaussianRBF

            return ClassRBF(
                var_fn=variance_f,
                amplitude=amplitude,
                length_scale=length_scale,
                power=alpha,
                feature_ndims=feature_ndims,
            )

        super().__init__(
            kernel_fn=kernel_fn,
            feature_ndims=1,
            mean_fn=mean_f,
            variance=variance,
            lengthscale=lengthscale,
            noise_var=noise_var,
            dtype=dtype,
        )
