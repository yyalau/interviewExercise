if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
from typing import Union, Any


class NLLBase:
    def __init__(self, model: Any, feature_ndims: int = 1, dtype: str = "float32"):
        '''
        Parameters:
        model: Any
            The model for training, and must have a method get_marginal_distribution. Using NLL as the loss function for optimization.
        feature_ndims: int
            The number of feature dimensions
        dtype: str
            The data type of the model. Should be either 'float32' or 'float64'
        '''
        self.sc(model, feature_ndims, dtype)
        self.model = model
        self.optimizer = tf.optimizers.Adam(
            learning_rate=0.005, beta_1=0.5, beta_2=0.99
        )
        self.dtype = dtype
        self.feature_ndims = feature_ndims
    
    def sc(self, model, feature_ndims: int, dtype: str) -> None:
        # assert model is not None, "model must be provided"
        assert model is not None, "model must be provided"
        assert isinstance(feature_ndims, int), "feature_ndims must be an integer"
        assert feature_ndims > 0, "feature_ndims must be a positive integer"
        assert isinstance(dtype, str), "dtype must be a string"
        assert dtype in ["float32", "float64"], "dtype must be 'float32' or 'float64'"
        
    def loss(self, x: Union[np.ndarray,tf.Tensor], y: Union[np.ndarray,tf.Tensor]) -> tf.Tensor:
        '''
        Using the negative log likelihood as the loss function for optimization. Requires the model to have a method get_marginal_distribution.
        Parameters:
        x: Union[np.ndarray,tf.Tensor]
            The input data
        y: Union[np.ndarray,tf.Tensor]
            The target data
        '''

        assert isinstance(x, (np.ndarray, tf.Tensor)), "x must be a tf.Tensor"
        assert isinstance(y, (np.ndarray, tf.Tensor)), "y must be a tf.Tensor"
        assert x.shape[0] == y.shape[0], "x and y must have the same length"

        return -self.model.get_marginal_distribution(x).log_prob(y) / x.shape[0]

    def optimize(self, x: Union[np.ndarray,tf.Tensor], y: Union[np.ndarray,tf.Tensor]) -> tf.Tensor:
        '''
        Parameters:
        x: Union[np.ndarray,tf.Tensor]
            The input data
        y: Union[np.ndarray,tf.Tensor]
            The target data
        '''

        assert isinstance(x, (np.ndarray, tf.Tensor)), "x must be a tf.Tensor"
        assert isinstance(y, (np.ndarray, tf.Tensor)), "y must be a tf.Tensor"
        assert x.shape[0] == y.shape[0], "x and y must have the same length"

        with tf.GradientTape() as tape:
            loss = self.loss(x, y)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def fit(
        self,
        x: Union[np.ndarray,tf.Tensor],
        y: Union[np.ndarray,tf.Tensor],
        n_restart: int =20,
        verbose: bool=False,
    ) -> tf.Tensor:
        '''
        Parameters:
        x: Union[np.ndarray,tf.Tensor]
            The input data
        y: Union[np.ndarray,tf.Tensor]
            The target data
        n_restart: int
            The number of restarts for optimization
        verbose: bool
            Whether to print the loss value / training details during optimization
        '''
        
        assert isinstance(x, (np.ndarray,tf.Tensor)), "x must be a tf.Tensor"
        assert isinstance(y, (np.ndarray,tf.Tensor)), "y must be a tf.Tensor"
        assert x.shape[0] == y.shape[0], "x and y must have the same length"
        assert isinstance(n_restart, int), "n_restart must be an integer"
        assert n_restart > 0, "n_restart must be a positive integer"
        assert isinstance(verbose, bool), "verbose must be a boolean"
        
        for i in range(n_restart):

            nll = self.optimize(x, y)
            if i % 5 == 0 and verbose:
                self.logging(i, nll)

        return self

    def logging(self, step: int, nll: tf.Tensor) -> None:
        '''
        Parameters:
        step: int
            The current step of the optimization
        nll: tf.Tensor
            The negative log likelihood value
        '''
        
        assert isinstance(nll, tf.Tensor), "nll must be a tf.Tensor"
        
        print("Step {}: NLL = {}".format(step, nll))