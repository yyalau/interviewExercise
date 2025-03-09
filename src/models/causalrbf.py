if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Callable, Union

tfb = tfp.bijectors
tfd = tfp.distributions

Quadratic = tfp.math.psd_kernels.ExponentiatedQuadratic
Gamma = tfp.math.psd_kernels.GammaExponential


class CausalRBF:
    def __init__(self, var_fn: Callable[[float], float]):
        '''
        Parameters:
        var_fn: Callable[[float], float]
            A function to calculate the variance of the kernel.
        '''
        self.var_fn = var_fn
        
    def _apply(self, x1: Union[tf.Tensor, np.ndarray], x2: Union[tf.Tensor, np.ndarray], example_ndims: int = 0) -> tf.Tensor:
        '''
        Apply the kernel function pairs of inputs.
        Parameters:
        x1: tf.Tensor or np.ndarray
            Input tensor.
        x2: tf.Tensor or np.ndarray
            Input tensor.
        example_ndims: int
            A python integer, the number of example dims in the inputs. In essence, this parameter controls how broadcasting of the kernel's batch shape with input batch shapes works. The kernel batch shape will be broadcast against everything to the left of the combined example and feature dimensions in the input shapes.
        '''
        assert isinstance(x1, (tf.Tensor, np.ndarray)), "x1 must be a tf.Tensor"
        assert type(x1) == type(x2), "x1 and x2 must have the same type. Got {} and {}".format(type(x1), type(x2))
        assert x1.shape == x2.shape, "x1 and x2 must have the same shape"
        assert isinstance(example_ndims, int), "example_ndims must be an integer"
        assert example_ndims >= 0, "example_ndims must be a non-negative integer"
                
        x1_diag = self.var_fn(x1)[..., None]
        x2_diag = self.var_fn(x2)[..., None]

        if example_ndims <= 1:
            x1_diag = x1_diag[..., 0]
            x2_diag = x2_diag[..., 0]

        result = tf.sqrt(x1_diag) * tf.sqrt(x2_diag)
        return result

    def _matrix(self, x1: Union[tf.Tensor, np.ndarray], x2: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
        '''
        Calculate the distance matrix between two tensors. 
        Parameters:
        x1: tf.Tensor or np.ndarray
            Input tensor.
        x2: tf.Tensor or np.ndarray
            Input tensor.
        '''
        
        assert isinstance(x1, (tf.Tensor, np.ndarray)), "x1 must be a tf.Tensor"
        assert type(x1) == type(x2), "x1 and x2 must have the same type. Got {} and {}".format(type(x1), type(x2))
        assert x1.shape == x2.shape, "x1 and x2 must have the same shape"

        x1_diag = self.var_fn(x1)[..., None]
        x2_diag = self.var_fn(x2)[..., None]

        if x1.shape[0] != 1 and x1_diag.shape[0] == 1:
            x1_diag = tf.repeat(x1_diag, x1.shape[0], axis=0)   
            x2_diag = tf.repeat(x2_diag, x2.shape[0], axis=0)
        
        return tf.sqrt(x1_diag) @ tf.sqrt(tf.transpose(x2_diag))

def create_kernel(BaseKernel):

    class WrapperRBF(CausalRBF, BaseKernel):
        def __init__(
            self,
            var_fn: Callable[[float], float],
            amplitude: Union[tfp.util.TransformedVariable, None] = None,
            length_scale: Union[tfp.util.TransformedVariable, None] = None,
            power: Union[float, None] = None,
            inverse_length_scale: Union[tfp.util.TransformedVariable, None] = None,
            feature_ndims: int = 1,
        ):
            '''
            Parameters:
            var_fn: Callable[[float], float]
                A function to calculate the variance of the kernel.
            amplitude: float
                The amplitude of the kernel.
            length_scale: float
                The length scale of the kernel.
            power: float
                The power of the kernel.
            inverse_length_scale: float
                The inverse length scale of the kernel.
            feature_ndims: int
                The number of feature dimensions.
            '''
            
            assert isinstance(var_fn, Callable), "var_fn must be a callable"
            assert amplitude is None or isinstance(amplitude, tfp.util.TransformedVariable), "amplitude must be a tfp.util.TransformedVariable, got {}".format(type(amplitude))
            assert length_scale is None or isinstance(length_scale, tfp.util.TransformedVariable), "length_scale must be a tfp.util.TransformedVariable, got {}".format(type(length_scale))
            assert power is None or isinstance(power, (int, float)), "power must be a number"
            assert inverse_length_scale is None or isinstance(inverse_length_scale, tfp.util.TransformedVariable), "inverse_length_scale must be a tfp.util.TransformedVariable, got {}".format(type(inverse_length_scale))
            assert isinstance(feature_ndims, int), "feature_ndims must be an integer"
            assert feature_ndims > 0, "feature_ndims must be a positive integer"
            
            CausalRBF.__init__(
                self,
                var_fn=var_fn,
            )
            
            if BaseKernel == Quadratic:            
                BaseKernel.__init__(
                    self,
                    amplitude=amplitude,
                    length_scale=length_scale,
                    inverse_length_scale=inverse_length_scale,
                    feature_ndims=feature_ndims,
                )
            else:
                BaseKernel.__init__(
                    self,
                    amplitude=amplitude,
                    length_scale=length_scale,
                    inverse_length_scale=inverse_length_scale,
                    power=power,
                    feature_ndims=feature_ndims,
                )
            
            

        def _apply(self, x1: Union[tf.Tensor, np.ndarray], x2: Union[tf.Tensor, np.ndarray], example_ndims: int = 0) -> tf.Tensor:
            '''
            Apply the kernel function pairs of inputs.
            Parameters:
            x1: tf.Tensor or np.ndarray
                Input tensor.
            x2: tf.Tensor or np.ndarray
                Input tensor.
            example_ndims: int
                A python integer, the number of example dims in the inputs. In essence, this parameter controls how broadcasting of the kernel's batch shape with input batch shapes works. The kernel batch shape will be broadcast against everything to the left of the combined example and feature dimensions in the input shapes.
            '''
            dist = BaseKernel._apply(self, x1, x2, example_ndims)
            adjust = CausalRBF._apply(self, x1, x2, example_ndims)
            result = tf.cast(adjust, dist.dtype) + dist
            return result

        def _matrix(self, x1: Union[tf.Tensor, np.ndarray], x2: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
            '''
            Construct the distance matrix between two tensors.
            Parameters:
            x1: tf.Tensor or np.ndarray
                Input tensor.
            x2: tf.Tensor or np.ndarray
                Input tensor.
            '''
            
            if self.feature_ndims == 1:
                dist = BaseKernel._matrix(self, x1, x2)
            else:
                dist = BaseKernel._matrix(self, x1[..., None], x2[..., None])

            adj = CausalRBF._matrix(self, x1, x2)
            result = tf.cast(adj, dist.dtype) + dist
            return result
    
    return WrapperRBF

GaussianRBF = create_kernel(Quadratic)
GammaRBF = create_kernel(Gamma)
