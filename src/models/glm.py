import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
from typing import Union

class GLMTanh:
    def __init__(self, A: float = 1., B: float = 1., trainable: bool = True, dtype: str = "float32", ):
        """
        Initialize the GLMTanh model.

        Parameters
        ----------
        dtype: str 
            Data type for the model variables.
        A: float
            Initial value for the A parameter.
        B: float
            Initial value for the B parameter.
        trainable: bool
            Whether the parameters A and B are trainable.
        """
        assert dtype in ["float32", "float64"], f"Invalid dtype: {dtype}. Must be 'float32' or 'float64'."
        assert isinstance(A, (float, int)), f"Invalid type for A: {type(A)}. Must be float or int."
        assert isinstance(B, (float, int)), f"Invalid type for B: {type(B)}. Must be float or int."
        assert isinstance(trainable, bool), f"Invalid type for trainable: {type(trainable)}. Must be bool."

        self.dtype = dtype
        
        self.g = tf.math.tanh
        
        self.A = tf.Variable(A, dtype=self.dtype, name="A", trainable=trainable)   
        self.B = tf.Variable(B, dtype=self.dtype, name="B", trainable=trainable)
        
        self.trainable_variables = [self.A, self.B] if trainable else []
        self.optimizer = tf.optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.99)
                
    def __call__(self, x: Union[np.ndarray, tf.Variable, tf.Tensor]) -> tf.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        x: tf.Tensor | np.ndarray | tf.Variable
            Input tensor.

        Returns
        ----------
        tf.Tensor
            Output tensor after applying the model.
        """
        assert isinstance(x, (tf.Tensor, tf.Variable, np.ndarray)), f"Invalid type for x: {type(x)}. Must be tf.Tensor."
        return self.A * self.g(self.B * x)
    
    
    @tf.function
    def optimize(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Perform one optimization step to minimize the mean squared error.

        Parameters
        ----------
        x: tf.Tensor
            Input tensor.
        y: tf.Tensor
            Target tensor.

        Returns
        ----------
        tf.Tensor
            Loss value after the optimization step.
        """
        
        assert isinstance(x, tf.Tensor), f"Invalid type for x: {type(x)}. Must be tf.Tensor."
        assert isinstance(y, tf.Tensor), f"Invalid type for y: {type(y)}. Must be tf.Tensor."
        
        with tf.GradientTape() as tape:
            y_hat = self(x)
            loss = tf.reduce_mean((y - y_hat)**2)
        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss
    
    def fit(self, x: tf.Tensor, y: tf.Tensor, n_restart: int = 10, verbose: bool = False) -> 'GLMTanh':
        """
        Fit the model to the given data.

        Parameters
        ----------
        x: tf.Tensor
            Input tensor.
        y: tf.Tensor
            Target tensor.
        n_restart: int
            Number of optimization steps.
        verbose: bool
            Whether to print progress during training.

        Returns
        ----------
        GLMTanh
            The fitted model.
        """
        
        assert isinstance(n_restart, int), f"Invalid type for n_restart: {type(n_restart)}. Must be int."
        assert isinstance(verbose, bool), f"Invalid type for verbose: {type(verbose)}. Must be bool."
        assert x.shape[0] == y.shape[0], f"The length of x and y must be the same. Got {x.shape[0]} and {y.shape[0]}."
        
        if self.trainable_variables == []:
            print("Model does not require fitting (len(trainable_variables) == 0)")
            return self
        
        for i in range(n_restart):
            mse = self.optimize(x, y)
            if i % 5 == 0 and verbose:
                print("Step {}: mse = {}".format(i, mse))
        
        if verbose:
            self.logging(mse)
        return self

    def logging(self, mse: tf.Tensor) -> None:
        """
        Log the final mean squared error and parameter values.

        Parameters
        ----------
        mse: tf.Tensor
            Final mean squared error.
        """
        print(f"Final mse = {mse}")
        print(f"A: {self.A.numpy()}, B: {self.B.numpy()}")
    

if __name__ == '__main__':
    # Generate synthetic data
    n = 3000
    t = tfd.Uniform(low=-np.pi, high=np.pi).sample([n, 1])
    x = tfd.Normal(loc=0., scale=20).sample([n, 1])
    y = 0.4 * tf.math.tanh(1.5 * x)  # + tfd.Normal(loc=0., scale=0.1).sample([n, 1])
    
    # Initialize and fit the model
    glm = GLMTanh(trainable=True)
    glm.fit(x, y, n_restart=500, verbose=True)