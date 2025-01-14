from models import GPRegression, GaussianRBF, GammaRBF
import tensorflow_probability as tfp

tfd = tfp.distributions


class BOModel(GPRegression):
    def __init__(
        self,
        es,
        target_var,
        mean_f,
        variance_f,
        variance=1.0,
        lengthscale=1.0,
        noise_var=1.0,
        alpha=2,
        beta=0.5,
        use_gamma_prior=True,
    ):
        self.es = es
        self.mean_f = mean_f
        self.variance_f = variance_f
        self.target_var = target_var

        def kernel_fn(amplitude, length_scale, feature_ndims):
            
            if use_gamma_prior:
                '''
                power = alpha, length_scale = beta
                '''
                return GammaRBF(
                    target_var=self.target_var,
                    var_fn=self.variance_f,
                    amplitude=amplitude,
                    length_scale=beta,
                    power = alpha,
                    feature_ndims=feature_ndims,
                )
            
            return GaussianRBF(
                target_var=self.target_var,
                var_fn=self.variance_f,
                amplitude=amplitude,
                length_scale=length_scale,
                feature_ndims=feature_ndims,
            )
        
        self.temp = kernel_fn(1, 1, len(es))
        if use_gamma_prior:
            tfd.Gamma(alpha, beta)

        super().__init__(
            kernel_fn,
            feature_ndims=len(es),
            mean_fn=mean_f,
            variance=variance,
            lengthscale=lengthscale,
            noise_var=noise_var,
        )
