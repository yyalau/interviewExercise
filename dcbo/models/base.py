if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

class NLLBase:
    def __init__(self, model, feature_ndims = 1, dtype = "float32"):
        self.model = model
        self.optimizer = tf.optimizers.Adam(learning_rate=0.005, beta_1=0.5, beta_2=0.99)
        self.dtype = dtype
        self.feature_ndims = feature_ndims
        
    def loss(self, y):
        return -self.model.log_prob(y,)
    
    @tf.function
    def optimize(
        self,
        y,
    ):
        with tf.GradientTape() as tape:
            loss = self.loss(y)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def fit(
        self,
        y,
        n_restart=10,
        verbose=False,
    ):
        for i in range(n_restart):
            nll = self.optimize(y)
            if i % 5 == 0 and verbose:
                print("Step {}: NLL = {}".format(i, nll))
        
        return nll