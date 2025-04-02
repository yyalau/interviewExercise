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

    def __init__(self, validate_args: bool = False, name: str = "ExpNorm"):
        """
        Initialize the ExpNorm bijector.

        Parameters
        ----------
        validate_args : bool
            Whether to validate input arguments.
        name : str
            Name of the bijector.
        """
        super().__init__(
            validate_args=validate_args, forward_min_event_ndims=0, name=name
        )
        self.sum = 1.0

    def _forward(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward transformation: normalizes the exponential of the input.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor.

        Returns
        -------
        tf.Tensor
            Transformed tensor.
        """
        self.sum = tf.reduce_sum(tf.exp(x))
        return tf.exp(x) / self.sum

    def _inverse(self, y: tf.Tensor) -> tf.Tensor:
        """
        Inverse transformation: computes the logarithm of the input.

        Parameters
        ----------
        y : tf.Tensor
            Input tensor.

        Returns
        -------
        tf.Tensor
            Inverse transformed tensor.
        """
        return tf.math.log(y * self.sum)

    def _inverse_log_det_jacobian(self, y: tf.Tensor) -> tf.Tensor:
        """
        Compute the log determinant of the Jacobian for the inverse transformation.

        Parameters
        ----------
        y : tf.Tensor
            Input tensor.

        Returns
        -------
        tf.Tensor
            Log determinant of the Jacobian.
        """
        return -self._forward_log_det_jacobian(self._inverse(y))

    def _forward_log_det_jacobian(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the log determinant of the Jacobian for the forward transformation.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor.

        Returns
        -------
        tf.Tensor
            Log determinant of the Jacobian.
        """
        return x


class MoG(NLLBase):
    def __init__(self, n_dist: int, link_fn: GLMTanh = None, dtype: str = "float32"):
        """
        Initialize the Mixture of Gaussians (MoG) model.

        Parameters
        ----------
        n_dist : int
            Number of Gaussian distributions in the mixture.
        link_fn : GLMTanh, optional
            Link function for the model. Defaults to None.
        dtype : str
            Data type for the model parameters. Defaults to "float32".
        """
        self.n_dist = n_dist
        self.dtype = dtype
        self.link_fn = link_fn

        self.w = tfp.util.TransformedVariable(
            np.arange(1, n_dist + 1), ExpNorm(), dtype=self.dtype, name="w", trainable=True
        )
        self.mean = tf.Variable([0.0] * self.n_dist, dtype=self.dtype, name="mean", trainable=True)
        self.variance = tfp.util.TransformedVariable(
            [1.0] * self.n_dist, tfb.Exp(), dtype=self.dtype, name="variance", trainable=True
        )

        model = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=self.w),
            components_distribution=tfd.Independent(
                distribution=tfd.Normal(loc=self.mean, scale=self.variance)
            ),
        )
        super().__init__(model=model)

    def loss(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Compute the negative log-likelihood loss.

        Parameters
        ----------
        x : tf.Tensor
            Input features.
        y : tf.Tensor
            Target values.

        Returns
        -------
        tf.Tensor
            Negative log-likelihood loss.
        """
        return -tf.reduce_mean(self.model.log_prob(y))

    def fit(self, x: tf.Tensor, y: tf.Tensor, n_restart: int = 100, verbose: bool = True):
        """
        Fit the MoG model to the data.

        Parameters
        ----------
        x : tf.Tensor
            Input features.
        y : tf.Tensor
            Target values.
        n_restart : int
            Number of restarts for optimization. Defaults to 100.
        verbose : bool
            Whether to print progress. Defaults to True.

        Returns
        -------
        tfd.Distribution
            Trained model.
        """


        if self.link_fn is not None:
            self.link_fn.fit(x, y, n_restart=n_restart, verbose=verbose)
            y = y - self.link_fn(x)

        nll = super().fit(x, y, n_restart=n_restart, verbose=verbose)
        if verbose:
            self.logging(nll)

        return self.model

    def logging(self, nll: float):
        """
        Log the final negative log-likelihood and model parameters.

        Parameters
        ----------
        nll : float
            Final negative log-likelihood value.
        """
        print(f"Final NLL = {nll}")
        print(f"Final w = {self.w._value().numpy()}")
        print(f"Final mean = {self.mean.numpy()}")
        print(f"Final variance = {self.variance._value().numpy()}")

        if self.link_fn is not None:
            print(f"A: {self.link_fn.A.numpy()}, B: {self.link_fn.B.numpy()}")

    def sample(self, n_samples: int, x: tf.Tensor = None) -> tf.Tensor:
        """
        Sample from the MoG model.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        x : tf.Tensor, optional
            Input features for the link function. Defaults to None.

        Returns
        -------
        tf.Tensor
            Generated samples.
        """
        result = self.model.sample(n_samples)

        if self.link_fn is not None and x is not None:
            result += self.link_fn(x)[..., 0]

        return result

    def prob(self, y: tf.Tensor, x: tf.Tensor = None) -> tf.Tensor:
        """
        Compute the probability of the given data under the MoG model.

        Parameters
        ----------
        y : tf.Tensor
            Target values.
        x : tf.Tensor, optional
            Input features for the link function. Defaults to None.

        Returns
        -------
        tf.Tensor
            Probability of the data.
        """
        if self.link_fn is not None and x is not None:
            y -= self.link_fn(x)

        return self.model.prob(y)