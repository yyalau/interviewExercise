import pytest
import tensorflow as tf
from models.bo_model import BOModel
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import numpy as np


def mean_function(x: float) -> float:
    return x

def variance_function(x: float) -> float:
    return x ** 2

@pytest.fixture
def bo_model():
    return BOModel(
        mean_f=mean_function,
        variance_f=variance_function,
        variance=1.0,
        lengthscale=0.5,
        noise_var=1.0,
        alpha=2,
        use_gamma_prior=True,
        dtype="float32"
    )

def test_bo_model_initialization(bo_model):
    assert bo_model.mean_fn == mean_function
    assert bo_model.kernel is not None
    assert bo_model.amplitude.numpy() == 1.0
    assert bo_model.length_scale.numpy() == 0.5
    assert bo_model.observation_noise_variance.numpy() == 1.0
    assert bo_model.dtype == "float32"
    assert isinstance(bo_model.model, tfd.GaussianProcess)

def test_bo_model_kernel_function(bo_model):
    x = np.array([[1.0], [2.0], [3.0]], dtype="float32")
    y = np.array([[1.0], [2.0], [3.0]], dtype="float32")
    answer = bo_model.kernel.apply(x, y)
    assert isinstance(bo_model.kernel, tfp.math.psd_kernels.AutoCompositeTensorPsdKernel)
    assert isinstance(answer, tf.Tensor)
    assert answer.shape == (3, 3)

def test_bo_model_mean_function(bo_model):
    x = 2.0
    mean = bo_model.mean_fn(x)
    assert mean == mean_function(x)
    assert isinstance(mean, float)
    assert not np.isnan(mean)
    assert not np.isinf(mean)

def test_bo_model_variance_function(bo_model):
    x = 2.0
    variance = bo_model.kernel.var_fn(x)
    assert variance == variance_function(x)
    assert isinstance(variance, float)
    assert not np.isnan(variance)
    assert not np.isinf(variance)