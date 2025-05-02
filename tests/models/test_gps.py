import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
import pytest
from src.models.gps import GPRegression



@pytest.fixture
def simple_gp():
    def rbf_kernel(amplitude, length_scale, feature_ndims):
        return tfp.math.psd_kernels.ExponentiatedQuadratic(
            amplitude=amplitude, length_scale=length_scale, feature_ndims=feature_ndims
        )
    return GPRegression(
        kernel_fn=rbf_kernel,
        feature_ndims=1,
        variance=1.0,
        lengthscale=1.0,
        noise_var=0.1,
        dtype="float32"
    )


def test_gp_fit_reduces_nll(simple_gp):
    x = np.linspace(-1, 1, 20).reshape(-1, 1).astype(np.float32)
    y = np.sin(x).reshape(-1)
    nll_before = simple_gp.loss(x, y).numpy() 
    simple_gp.fit(x, y, n_restart=3, verbose=False)
    nll_after = simple_gp.loss(x, y).numpy() 
    assert nll_after < nll_before, "NLL did not decrease after fitting the model. nll_before: {}, nll_after: {}".format(nll_before, nll_after)

def test_gp_predict_shapes(simple_gp):
    x = np.linspace(-1, 1, 10).reshape(-1, 1).astype(np.float32)
    y = np.sin(x).reshape(-1)
    simple_gp.fit(x, y, n_restart=2)
    x_test = np.linspace(-2, 2, 5).reshape(-1, 1).astype(np.float32)
    mean, var = simple_gp.predict(x_test)
    assert mean.shape == (5,)
    assert var.shape == (5,)


def test_gp_predict_shapes_2d(simple_gp):

    x = np.linspace(-1, 1, 20).reshape(-1, 2).astype(np.float32)
    y = np.sin(x[:, 1]) + np.cos(x[:, 0])
    print(y.shape)
    simple_gp.fit(x, y, n_restart=2)
    
    x_test = np.linspace(-2, 2, 10).reshape(-1, 2).astype(np.float32)
    mean, var = simple_gp.predict(x_test)
    assert mean.shape == (5,)
    assert var.shape == (5,)
