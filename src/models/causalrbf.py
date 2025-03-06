if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions

Quadratic = tfp.math.psd_kernels.ExponentiatedQuadratic
Gamma = tfp.math.psd_kernels.GammaExponential

from tensorflow.python.framework.ops import SymbolicTensor

class CausalRBF:
    def __init__(self, target_var, var_fn):
        # super().__init__(
        #     amplitude= amplitude,
        #     length_scale= length_scale,
        #     power = power,
        #     inverse_length_scale= inverse_length_scale,
        #     feature_ndims= feature_ndims,
        # )
        """
        TODO
        ARD feature is not implemented
        https://gpy.readthedocs.io/en/deploy/_modules/GPy/kern/src/stationary.html#Stationary.K
        """

        self.X = None
        self.Y = None
        self.var_fn = var_fn
        self.target_var = target_var
        # self.dtypec = dtype
        
    def _apply(self, x1, x2, example_ndims=0):
        # import ipdb; ipdb.set_trace()
        x1_diag = self.var_fn(x1)[..., None]
        x2_diag = self.var_fn(x2)[..., None]

        if example_ndims <= 1:
            x1_diag = x1_diag[..., 0]
            x2_diag = x2_diag[..., 0]

        result = tf.sqrt(x1_diag) * tf.sqrt(x2_diag)
        return result

    def _matrix(self, x1, x2):
        
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
            target_var,
            var_fn,
            amplitude=1,
            length_scale=None,
            power=None,
            inverse_length_scale=None,
            feature_ndims=1,
            dtype="float32",
        ):
            CausalRBF.__init__(
                self,
                target_var=target_var,
                var_fn=var_fn,
            )
            
            BaseKernel.__init__(
                self,
                amplitude=amplitude,
                length_scale=length_scale,
                power=power,
                inverse_length_scale=inverse_length_scale,
                feature_ndims=feature_ndims,
            )
            

        def _apply(self, x1, x2, example_ndims=0):
            dist = BaseKernel._apply(self, x1, x2, example_ndims)
            adjust = CausalRBF._apply(self, x1, x2, example_ndims)
            result = tf.cast(adjust, dist.dtype) + dist
            return result

        def _matrix(self, x1, x2):

            if self.feature_ndims == 1:
                # import ipdb; ipdb.set_trace()
                dist = BaseKernel._matrix(self, x1, x2)
            else:
                dist = BaseKernel._matrix(self, x1[..., None], x2[..., None])

            # if not isinstance(dist, SymbolicTensor) and np.isnan(dist.numpy()).any():
            #     import ipdb; ipdb.set_trace()      


            adj = CausalRBF._matrix(self, x1, x2)
            result = tf.cast(adj, dist.dtype) + dist
            return result
    
    return WrapperRBF

GaussianRBF = create_kernel(Quadratic)
GammaRBF = create_kernel(Gamma)


# class GammaRBF(CausalRBF, Gamma):
#     def __init__(
#         self,
#         target_var,
#         var_fn,
#         amplitude=1,
#         length_scale=None,
#         power=None,
#         inverse_length_scale=None,
#         feature_ndims=1,
#         dtype="float32",
#     ):
#         CausalRBF.__init__(
#             self,
#             target_var=target_var,
#             var_fn=var_fn,
#         )

#         Gamma.__init__(
#             self,
#             amplitude=amplitude,
#             length_scale=length_scale,
#             power=power,
#             inverse_length_scale=inverse_length_scale,
#             feature_ndims=feature_ndims,
#         )

#     def _apply(self, x1, x2, example_ndims=0):
#         dist = Gamma._apply(self, x1, x2, example_ndims)
#         adjust = CausalRBF._apply(self, x1, x2, example_ndims)
#         result = tf.cast(adjust, dist.dtype) + dist
#         return result

#     def _matrix(self, x1, x2):

#         if self.feature_ndims == 1:
#             dist = Gamma._matrix(self, x1, x2)
#         else:
#             dist = Gamma._matrix(self, x1[..., None], x2[..., None])

#         adj = CausalRBF._matrix(self, x1, x2)
#         result = tf.cast(adj, dist.dtype) + dist
#         return result

