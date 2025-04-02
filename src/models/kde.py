
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np
from typing import Callable, Union

class KernelDensity:
    def __init__(
        self,
        kernel: Callable,
        n_bins: int = 100,
        bandwidth: Union[float, str] = 1.0,
        dtype: str = "float32",
    ):
        """
        Initialize the KernelDensity estimator.

        Parameters
        ----------
        kernel: Callable
            Kernel function (e.g., `tfd.Normal`).
        n_bins: int
            Number of bins for the density estimation. Defaults to 100.
        bandwidth: Union[float, str]
            Bandwidth for the kernel. Can be a float or one of ["scott", "silverman"]. Defaults to 1.0.
        dtype: str
            Data type for computations. Defaults to "float32".
        """
        self.kernel = kernel
        self.n_bins = n_bins
        self.bandwidth = bandwidth
        self.dtype = dtype

    def _set_bandwidth(self, x: tf.Tensor) -> None:
        """
        Set the bandwidth for the kernel density estimation.

        Parameters
        ----------
        x: tf.Tensor
            Input data tensor.

        Raises
        ------
        ValueError
            If the bandwidth string is invalid.
        """
        if isinstance(self.bandwidth, str):
            if self.bandwidth not in ["scott", "silverman"]:
                raise ValueError("Invalid bandwidth. Must be 'scott' or 'silverman'")

            if self.bandwidth == "scott":
                self.bandwidth_ = x.shape[0] ** (-1 / (x.shape[1] + 4))
            elif self.bandwidth == "silverman":
                self.bandwidth_ = (x.shape[0] * (x.shape[1] + 2) / 4) ** (-1 / (x.shape[1] + 4))
        else:
            self.bandwidth_ = self.bandwidth

    def fit(self, x: tf.Tensor) -> "KernelDensity":
        """
        Fit the KernelDensity model to the input data.

        Parameters
        ----------
        x: tf.Tensor
            Input data tensor.

        Returns
        -------
        KernelDensity
            The fitted KernelDensity instance.
        """
        self._set_bandwidth(x)

        # https://astroviking.github.io/ba-thesis/tensorflowImplementation.html
        probs = tf.ones(x.shape[0], dtype=self.dtype) / x.shape[0]
        f = lambda x: tfd.Independent(self.kernel(loc=x, scale=self.bandwidth_))

        self.kde = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=probs),
            components_distribution=f(x),
        )

        return self

    def sample(self, n_samples: int) -> tf.Tensor:
        """
        Generate samples from the fitted KernelDensity model.

        Parameters
        ----------
        n_samples: int
            Number of samples to generate.

        Returns
        -------
        tf.Tensor
            Generated samples.
        """
        return self.kde.sample(n_samples)

if __name__ == "__main__":

    tf.random.set_seed(123)
    np.random.seed(123)

    X = np.random.rand(12, 1)
    Y = np.random.rand(12)
    kde = KernelDensity(tfd.Normal).fit(X)