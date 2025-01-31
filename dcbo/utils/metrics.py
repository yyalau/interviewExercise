import tensorflow as tf
    
class BayesFactor:
    '''
    H_0 = p(y); H_1 = p(y|X)
    BF = P(D | H_0) / P(D | H_1)
    BF = P(H_0 | D) * P(H_1) / P(H_1 | D) * P(H_0)
    
    '''
    def __call__(self, D_Int, y, m0, m1, x = None):
        
        y_hat = 1
        if D_Int.dataX is not None and D_Int.dataY is not None:
            yhat0 = m0.prob(D_Int.dataY)
            yhat1 = m1.prob(D_Int.dataY)
            y_hat = tf.math.reduce_prod(yhat0 / yhat1, 0)
            
        y0 = m0.prob(y)
        y1 = m1.prob(y, x = x)
        
        return y0 / y1 * y_hat

class PDC:
    
    def __init__(self, beta, k0, k1, feature_dim = 1, dtype="float32"):
        self.beta = beta
        self.k0 = k0
        self.k1 = k1
        self.bf = BayesFactor()
        
        self.x  = tf.Variable(tf.zeros((feature_dim,1), dtype = dtype), dtype=dtype, name="mean",  trainable=True)        

        self.optimizer = tf.optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.99)
    
    def __call__(self, prob_H0, D_Int, m0, m1, n_samples = 1):
        
        prob_H1 = 1 - prob_H0
        y0 = tf.convert_to_tensor(m0.sample(n_samples, x = None))
        y1 = tf.convert_to_tensor(m1.sample(n_samples, x = self.x)) # x is given here
        
        def p(k):
            return tf.reduce_mean(tf.exp(-1. / self.beta * tf.nn.relu(k)))
        
        result =  p(self.k0 - self.bf(D_Int, y0, m0, m1))*prob_H0
        result += p(self.bf(D_Int, y1, m0, m1, x= self.x) - self.k1)*prob_H1
        
        return result
    
    @tf.function
    def optimize(self, prob_H0, D_Int, m0, m1, n_samples):
        
        with tf.GradientTape() as tape:
            loss = -self(prob_H0, D_Int, m0, m1, n_samples)
        
        grads = tape.gradient(loss, [self.x])
        self.optimizer.apply_gradients(zip(grads, [self.x]))
        return -loss
    
    
    def fit(self, prob_H0, D_Int, m0, m1, n_samples = 200, n_restart=300, verbose=True):
        
        for i in range(n_restart):
            pdc = self.optimize(prob_H0, D_Int, m0, m1, n_samples)
            if i % 5 == 0 and verbose:
                print("Step {}: pdc = {}".format(i, pdc))
        
        if verbose: self.logging(pdc)
                    
        return self
    
    def get_xopt(self):
        return self.x.numpy()
    
    def logging(self, pdc):
        print(f"Final PDC (max) = {pdc}")
        print(f"Final x = {self.x.numpy()}")