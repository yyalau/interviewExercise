if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

if __name__ != "__main__":
    from .base import NLLBase
    from .glm import GLMTanh
else:
    from base import NLLBase
    from glm import GLMTanh
    
tfb = tfp.bijectors
tfd = tfp.distributions

class ExpNorm(tfb.Bijector):

    def __init__(self, validate_args=False, name="ExpNorm"):
        super().__init__(
            validate_args=validate_args, forward_min_event_ndims=0, name=name
        )
        self.sum = 1.0

    def _forward(self, x):

        self.sum = tf.reduce_sum(tf.exp(x))

        return tf.exp(x) / self.sum

    def _inverse(self, y):

        return tf.math.log(y * self.sum)

    def _inverse_log_det_jacobian(self, y):
        return -self._forward_log_det_jacobian(self._inverse(y))

    def _forward_log_det_jacobian(self, x):
        # Notice that we needn't do any reducing, even when`event_ndims > 0`.
        # The base Bijector class will handle reducing for us; it knows how
        # to do so because we called `super` `__init__` with
        # `forward_min_event_ndims = 0`.
        return x


class MoG(NLLBase):
    def __init__(
        self,
        n_dist,
        link_fn = None,
        dtype="float32",
    ):
        self.X = None
        self.Y = None
        self.n_dist = n_dist
        self.dtype = dtype
        
        self.link_fn = link_fn
        
        self.w = tfp.util.TransformedVariable(
            np.arange(1, n_dist+1) , ExpNorm(), dtype=self.dtype, name="w", trainable = True
        )
        self.mean  = tf.Variable([0.]*self.n_dist, dtype=self.dtype, name="mean",  trainable=True)        
        self.variance = tfp.util.TransformedVariable( [1.]*self.n_dist, tfb.Exp(), dtype=self.dtype, name="variance",  trainable=True)
        
        model = tfd.MixtureSameFamily(
            mixture_distribution = tfd.Categorical(probs=self.w),
            components_distribution = tfd.Independent(
              distribution = tfd.Normal(loc = self.mean, scale = self.variance)
            )
        )
        super().__init__(model = model)
    
    def loss(self, y):
        return - tf.reduce_mean(self.model.log_prob(y))

    def fit(self, x, y, n_restart=100, verbose=True):
        self.X = x
        self.Y = y
        
        if self.link_fn is not None:
            self.link_fn.fit(x, y, n_restart=n_restart, verbose=verbose)
            y = y - self.link_fn(x)
        
        nll = super().fit(y, n_restart=n_restart, verbose=verbose)      
        if verbose: self.logging(nll)
                    
        return self.model
    
    def logging(self, nll):
        print(f"Final NLL = {nll}")
        print(f"Final w = {self.w._value().numpy()}")
        print(f"Final mean = {self.mean.numpy()}")
        print(f"Final variance = {self.variance._value().numpy()}")
        
        if self.link_fn is not None:
            print(f"A: {self.link_fn.A.numpy()}, B: {self.link_fn.B.numpy()}")

    def sample(self, n_samples, x = None):
        result =  self.model.sample(n_samples)
        
        if self.link_fn is not None and x is not None:
            result += self.link_fn(x)[..., 0]
        
        return result

    def prob(self, y, x =None):
        
        if self.link_fn is not None and x is not None:
            y -= self.link_fn(x)
        
        return self.model.prob(y)

if __name__ == "__main__":
    n = 3000
    t = tfd.Uniform(low=-np.pi, high=np.pi).sample([n, 1])
    x = tfd.Normal(loc=0., scale=20).sample([n, 1])
    y = 0.4*tf.math.tanh(0.9*x)
        
    mog = MoG(n_dist=2, dtype=x.dtype, link_fn=GLMTanh(trainable=True))
    mog.fit( x, y, n_restart=500, verbose=True)
    
    k = 6
    xx = tfd.Normal(loc=0., scale=1).sample(k)
    print(mog.sample(k, xx))
    print(0.4*tf.math.tanh(0.9*xx))