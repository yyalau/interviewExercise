import pytest
import numpy as np
import tensorflow as tf
from models.causalrbf import CausalRBF, GaussianRBF, GammaRBF

# filepath: /home/dycpu4_4tssd/yyalau/github/interviewExercise/tests/models/test_causalrbf.py


def variance_function(x: float) -> float:
    return x ** 2

@pytest.fixture
def causal_rbf():
    return CausalRBF(var_fn=variance_function)

@pytest.fixture
def gaussian_rbf():
    return GaussianRBF(var_fn=variance_function)

@pytest.fixture
def gamma_rbf():
    return GammaRBF(var_fn=variance_function)

def test_causalrbf_apply(causal_rbf):
    x1 = tf.convert_to_tensor([1.0, 2.0, 3.0])
    x2 = tf.convert_to_tensor([1.0, 2.0, 3.0])
    result = causal_rbf._apply(x1, x2)
    expected = tf.sqrt(variance_function(x1)) * tf.sqrt(variance_function(x2))
    tf.debugging.assert_near(result, expected)

@pytest.mark.parametrize(
    "rbf",
    [
        CausalRBF(var_fn=variance_function),
        GaussianRBF(var_fn=variance_function),
        GammaRBF(var_fn=variance_function),
    ],
)
def test_causalrbf_apply_tf(rbf):
    x1 = tf.convert_to_tensor([1.0, 2.0, 3.0])
    x2 = tf.convert_to_tensor([1.0, 2.0, 3.0])
    result = rbf._apply(x1, x2)
    expected = tf.sqrt(variance_function(x1)) * tf.sqrt(variance_function(x2))
    tf.debugging.assert_near(result, expected)

    assert isinstance(result, tf.Tensor)
    assert result.shape == (3,)
    assert tf.math.is_nan(result).numpy().all() == False

@pytest.mark.parametrize(
    "rbf",
    [
        CausalRBF(var_fn=variance_function),
        GaussianRBF(var_fn=variance_function),
        GammaRBF(var_fn=variance_function),
    ],
)
def test_causalrbf_apply_np(rbf):
    x1 = np.array([1.0, 2.0, 3.0])
    x2 = np.array([1.0, 2.0, 3.0])
    result = rbf._apply(x1, x2, )
        
    assert isinstance(result, tf.Tensor)
    assert result.shape == (3,)
    assert tf.math.is_nan(result).numpy().all() == False

@pytest.mark.parametrize(
    "rbf",
    [
        CausalRBF(var_fn=variance_function),
        GaussianRBF(var_fn=variance_function),
        GammaRBF(var_fn=variance_function),
    ],
)
def test_causalrbf_invalid_inputs(rbf):
    x1 = tf.convert_to_tensor([1.0, 2.0, 3.0])
    x2 = tf.convert_to_tensor([1.0, 2.0])
    with pytest.raises(AssertionError):
        rbf._apply(x1, x2)

    x1 = np.array([1.0, 2.0, 3.0])
    x2 = tf.convert_to_tensor([1.0, 2.0, 3.0])
    with pytest.raises(AssertionError):
        rbf._apply(x1, x2)

    x1 = tf.convert_to_tensor([1.0, 2.0, 3.0])
    x2 = tf.convert_to_tensor([1.0, 2.0, 3.0])
    with pytest.raises(AssertionError):
        rbf._apply(x1, x2, example_ndims=-1)

@pytest.mark.parametrize(
    "rbf",
    [
        CausalRBF(var_fn=variance_function),
        GaussianRBF(var_fn=variance_function),
        GammaRBF(var_fn=variance_function),
    ],
)
def test_causalrbf_matrix_tf(rbf):
    x1 = tf.convert_to_tensor([[1.0], [2.0], [3.0],])
    x2 = tf.convert_to_tensor([[1.0], [2.0], [3.0],])
    result = rbf._matrix(x1, x2)
    expected = tf.sqrt(variance_function(x1)) @ tf.sqrt(tf.transpose(variance_function(x2)))
    # tf.debugging.assert_near(result, expected)
    
    print(result)
    print(expected)

    assert isinstance(result, tf.Tensor)
    assert result.shape == (3, 3)
    assert tf.math.is_nan(result).numpy().all() == False

@pytest.mark.parametrize(
    "rbf",
    [
        CausalRBF(var_fn=variance_function),
        GaussianRBF(var_fn=variance_function),
        GammaRBF(var_fn=variance_function),
    ],
)
def test_causalrbf_matrix_np(rbf):
    x1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = rbf._matrix(x1, x2)
    expected = tf.sqrt(variance_function(x1)) @ tf.sqrt(tf.transpose(variance_function(x2)))
    tf.debugging.assert_near(result, expected)

    assert isinstance(result, tf.Tensor)
    assert result.shape == (2, 2)
    assert tf.math.is_nan(result).numpy().all() == False
# generate matrix testcases