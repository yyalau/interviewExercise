import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels


class GPRegression:
    def __init__(self):
        self.X = None
        self.Y = None
        self.gp = None
        

    def fit(self, x, y,    
        lengthscale=1.0,
        variance=1.0,
        noise_var=1.0,
        ard=False,
        n_restart=10,
        seed: int = 0,
        verbose=False,):
        
        self.X = x
        self.Y = y
        
        # Define a kernel with trainable parameters. Note we use TransformedVariable
        # to apply a positivity constraint.
        amplitude = tfp.util.TransformedVariable(
        variance, tfb.Exp(), dtype=tf.float64, name='amplitude')
        
        # ard = true is not implemented
        length_scale = tfp.util.TransformedVariable(
        lengthscale, tfb.Exp(), dtype=tf.float64, name='length_scale')

        feature_ndims = x.shape[1]
        kernel = psd_kernels.ExponentiatedQuadratic(amplitude, length_scale, feature_ndims=feature_ndims)

        observation_noise_variance = tfp.util.TransformedVariable(
            noise_var, tfb.Exp(), dtype=tf.float64,name='observation_noise_variance')
        

        # We'll use an unconditioned GP to train the kernel parameters.
        self.gp = tfd.GaussianProcess(
            kernel=kernel,
            index_points=x,
            observation_noise_variance=observation_noise_variance)

        optimizer = tf.optimizers.Adam(learning_rate=.05, beta_1=.5, beta_2=.99)

        @tf.function
        def optimize():
            with tf.GradientTape() as tape:
                loss = -self.gp.log_prob(y,)
            grads = tape.gradient(loss, self.gp.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.gp.trainable_variables))
            return loss
        
        
        for i in range(n_restart):
            neg_log_likelihood_ = optimize()
            if i % 50 == 0 and verbose:
                print("Step {}: NLL = {}".format(i, neg_log_likelihood_))

        if verbose:
            print("Final NLL = {}".format(neg_log_likelihood_))
            
            print("Trained parameters:")
            print("amplitude: {}".format(amplitude._value().numpy()))
            print("length_scale: {}".format(length_scale._value().numpy()))
            print(
                "observation_noise_variance: {}".format(
                    observation_noise_variance._value().numpy()
                )
            )

        return self
    
    def gprm(self):
        return lambda inputs: tfd.GaussianProcessRegressionModel(
            kernel=self.gp.kernel,
            index_points=inputs,
            observation_index_points=self.X,
            observations=self.Y,
            observation_noise_variance=self.gp.observation_noise_variance
        )

    def predict(self, x):
        gprm_fit = self.gprm()(x[...,None])
        # import ipdb; ipdb.set_trace()
        return gprm_fit.mean().numpy().reshape(-1), gprm_fit.variance().numpy().reshape(-1)

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