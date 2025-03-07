if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np

class GLMTanh:
    def __init__(self, dtype = "float32", A = 1., B = 1., trainable = True):
        self.dtype = dtype
        
        self.g = tf.math.tanh
        
        self.A = tf.Variable(A, dtype=self.dtype, name="A",  trainable=trainable)   
        self.B = tf.Variable(B, dtype=self.dtype, name="B",  trainable=trainable)
        
        self.trainable_variables = [self.A, self.B] if trainable else []
        self.optimizer = tf.optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.99)
                
    def __call__(self, x):
        return self.A*self.g(self.B * x)
    
    
    @tf.function
    def optimize(self, x, y):
        with tf.GradientTape() as tape:
            y_hat = self(x)
            loss = tf.reduce_mean((y - y_hat)**2)
        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss
    
    def fit(self, x, y, n_restart=10, verbose=False):
        
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

    def logging(self, mse):
        print(f"Final mse = {mse}")
        print(f"A: {self.A.numpy()}, B: {self.B.numpy()}")
    

if __name__ == '__main__':
    n = 3000
    t = tfd.Uniform(low=-np.pi, high=np.pi).sample([n, 1])
    x = tfd.Normal(loc=0., scale=20).sample([n, 1])
    y = 0.4*tf.math.tanh(1.5*x)    #+ tfd.Normal(loc=0., scale=0.1).sample([n, 1])
    
    glm = GLMTanh(trainable=True)
    glm.fit(x, y, n_restart=500, verbose=True)