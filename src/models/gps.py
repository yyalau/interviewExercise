import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from .base import NLLBase


tfb = tfp.bijectors
tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels


class GPRegression(NLLBase):
    def __init__(
        self,
        kernel_fn,
        feature_ndims,
        mean_fn=None,
        variance=1.0,
        lengthscale=1.0,
        noise_var=1.0,
        dtype = "float32",
    ):
        self.X = None
        self.Y = None
        self.mean_fn = mean_fn
        self.dtype = dtype


        self.amplitude = tfp.util.TransformedVariable(
            variance, tfb.Exp(), dtype=dtype, name="amplitude"
        )
        # self.amplitude = variance
        
        self.length_scale = tfp.util.TransformedVariable(
            lengthscale, tfb.Exp(), dtype=dtype, name="length_scale"
        )
        # self.length_scale = lengthscale
        
        self.observation_noise_variance = tfp.util.TransformedVariable(
            noise_var, tfb.Exp(), dtype=dtype, name="observation_noise_variance"
        )
        # self.observation_noise_variance = noise_var

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
        
                
        super().__init__(model = model, feature_ndims=feature_ndims, dtype=dtype)
        
        
    # def gprm(self):
    #     return lambda inputs: tfd.GaussianProcessRegressionModel(
    #         kernel=self.model.kernel,
    #         index_points=inputs,
    #         observation_index_points=self.X,
    #         observations=self.Y,
    #         observation_noise_variance=self.model.observation_noise_variance,
    #     )
        

        # https://towardsdatascience.com/gaussian-process-models-7ebce1feb83d

    # @tf.function
    # def optimize(
    #     self,
    #     y,
    # ):
    #     with tf.GradientTape() as tape:
    #         loss = -self.model.log_prob(y,)

    #     grads = tape.gradient(loss, self.model.trainable_variables)
    #     self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    #     return loss

    def fit(
        self,
        x,
        y,
        ard=False,
        n_restart=10,
        seed: int = 0,
        verbose=False,
    ):
        self.X = x
        self.Y = y
        
        # We'll use an unconditioned GP to train the kernel parameters.        

        nll = super().fit(x, y, n_restart=n_restart, verbose=verbose)      
        
        if verbose:
            self.logging(nll)

        return self

    def logging(self, nll):
        print("Final NLL = {}".format(nll))
        print("Trained parameters:")
        print("amplitude: {}".format(self.amplitude._value().numpy()))
        print("length_scale: {}".format(self.length_scale._value().numpy()))
        print(
            "observation_noise_variance: {}".format(
                self.observation_noise_variance._value().numpy()
            )
        )

    # def gprm(self, inputs):
    #     return 

    def predict(self, x):
        if self.feature_ndims == 1 and x.ndim == 1 or self.feature_ndims > 1 and x.ndim == 2:
            x = x[...,None]
        # import ipdb; ipdb.set_trace()
        gp_fit = self.model.get_marginal_distribution(x)   
        # return tf.reshape(gprm_fit.mean(), (x.shape[0], -1)), tf.reshape(gprm_fit.variance(), (x.shape[0], -1))
        return gp_fit.mean(), gp_fit.variance()


# if __name__ == "__main__":
#     import tensorflow as tf
#     physical_devices = tf.config.list_physical_devices('GPU')
#     tf.config.set_visible_devices(physical_devices[5], 'GPU')

#     tf.random.set_seed(123)
#     np.random.seed(123)

#     X = np.random.rand(12,2,1)
#     Y = np.random.rand(12)
#     gp = fit_gp(X, Y)


#     X2 = np.random.rand(12,2,1)
#     Y2 = predict(gp, X2)
#     print(Y2)

# import matplotlib.pyplot as plt
# plt.scatter(np.squeeze(observation_index_points), y,)
# plt.plot(np.stack([index_points[:, 0]]*10).T, samples.T, c='r', alpha=.2)
