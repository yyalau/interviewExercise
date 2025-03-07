from . import GPRegression, GaussianRBF, GammaRBF
import tensorflow_probability as tfp

tfd = tfp.distributions


class BOModel(GPRegression):
    def __init__(
        self,
        es,
        target_var,
        mean_f,
        variance_f,
        nll_variance_f = None,
        variance=1.0,
        lengthscale=1.0,
        noise_var=1.0,
        alpha=2,
        beta=0.5,
        use_gamma_prior=True,
        dtype="float32",
    ):
        self.es = es
        self.target_var = target_var

        def kernel_fn(amplitude, length_scale, feature_ndims):
            ClassRBF = GammaRBF if use_gamma_prior else GaussianRBF
            '''
            power = alpha, length_scale = beta
            '''
            return ClassRBF(
                target_var=self.target_var,
                var_fn=variance_f if nll_variance_f is None else nll_variance_f,
                amplitude=amplitude,
                length_scale=beta,
                power = alpha,
                feature_ndims=feature_ndims,
            )
            


        super().__init__(
            kernel_fn = kernel_fn,
            feature_ndims=1,
            mean_fn=mean_f,
            variance=variance,
            lengthscale=lengthscale,
            noise_var=noise_var,
            dtype = dtype,
        )
