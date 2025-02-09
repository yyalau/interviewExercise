import tensorflow as tf


class BayesFactor:
    """
    H_0 = p(y); H_1 = p(y|X)
    BF = P(D | H_0) / P(D | H_1)
    BF = P(H_0 | D) * P(H_1) / P(H_1 | D) * P(H_0)

    """

    def __call__(self, m0, m1, dataY, dataX=None, reduce=True):
        if dataY is None:
            return 1

        y0 = m0.prob(dataY)
        y1 = m1.prob(dataY, x=dataX)

        if reduce:
            return tf.reduce_prod(y0 / y1)
        return y0 / y1


class CriterionBase:
    def __init__(self, feature_dim, dtype, name, task="max"):
        self.feature_dim = feature_dim
        self.dtype = dtype
        self.x = tf.Variable(
            tf.zeros((feature_dim, 1), dtype=dtype),
            dtype=dtype,
            name="x",
            trainable=True,
        )
        self.optimizer = tf.optimizers.Adam(
            learning_rate=0.005, beta_1=0.9, beta_2=0.99
        )
        self.name = name
        self.task = task
        

    @tf.function
    def optimize(self, prob_H0, D_Int, m0, m1, n_samples):

        with tf.GradientTape() as tape:
            loss = self(prob_H0, D_Int, m0, m1, n_samples)
            if self.task == "max":
                loss = -loss

        grads = tape.gradient(loss, [self.x])
        self.optimizer.apply_gradients(zip(grads, [self.x]))
        return loss

    def fit(self, prob_H0, D_Int, m0, m1, n_samples=200, n_restart=300, verbose=True):

        for i in range(n_restart):
            pdc = self.optimize(prob_H0, D_Int, m0, m1, n_samples)
            if i % 5 == 0 and verbose:
                print(f"Step {i}: {self.name} = ({self.task}) {pdc}")
        if verbose:
            self.logging(pdc)

        return self

    def get_xopt(self):
        return self.x.numpy()

    def logging(self, score):
        print(f"Final {self.name} = {score}")
        print(f"Final x = {self.x.numpy()}")


class PDC(CriterionBase):

    def __init__(self, beta, k0, k1, feature_dim=1, dtype="float32"):
        super().__init__(feature_dim, dtype, "pdc", "max")
        self.beta = beta
        self.k0 = k0
        self.k1 = k1
        self.bf = BayesFactor()

    def __call__(self, prob_H0, D_Int, m0, m1, n_samples=1):

        prob_H1 = 1 - prob_H0
        y0 = tf.convert_to_tensor(m0.sample(n_samples, x=None))
        y1 = tf.convert_to_tensor(m1.sample(n_samples, x=self.x))  # x is given here

        def p(k):
            return tf.reduce_mean(tf.exp(-1.0 / self.beta * tf.nn.relu(k)))

        d_int_bf = self.bf(m0, m1, dataY=D_Int.dataY, dataX=D_Int.dataX)

        result = (
            p(self.k0 - d_int_bf * self.bf(m0, m1, dataY=y0, dataX=None, reduce=False))
            * prob_H0
        )
        result += (
            p(d_int_bf * self.bf(m0, m1, dataY=y1, dataX=self.x) - self.k1) * prob_H1
        )

        return result


class InfoGain(CriterionBase):
    def __init__(self, feature_dim=1, dtype="float32"):
        super().__init__(feature_dim, dtype, "infogain", "max")
        self.bf = BayesFactor()

    def subinfo(self, prob_H0, dataY, dataX, m0, m1):
        c = (bf_d := self.bf(m0, m1, dataY, dataX=dataX)) * prob_H0 + (
            prob_H1 := 1 - prob_H0
        )

        result = m0.prob(dataY) * prob_H0 * tf.math.log(bf_d / c)
        result += m1.prob(dataY, x=dataX) * prob_H1 * tf.math.log(1 / c)

        return tf.reduce_sum(result)

    def __call__(self, prob_H0, D_Int, m0, m1, n_samples=1):
        y_new = tf.reshape(m1.sample(n_samples, x=self.x), (-1, self.feature_dim))

        def concat(x, y):
            return tf.concat([x, y], 0) if x is not None else y

        dataY = concat(D_Int.dataY, y_new)
        dataX = concat(D_Int.dataX, tf.repeat(self.x, n_samples, axis=0))

        I_or = self.subinfo(prob_H0, dataY, dataX, m0, m1)
        I_new = self.subinfo(prob_H0, y_new, self.x, m0, m1)

        return I_or - I_new
