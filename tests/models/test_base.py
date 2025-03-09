import pytest
import tensorflow as tf
import numpy as np
from models.base import NLLBase
import tensorflow_probability as tfp


class DummyModel:
    def __init__(self, dtype = "float32"):
        # self.trainable_variables = []
        self.variance = tf.Variable(1.0, dtype=dtype, name="variance", trainable=True)  
        self.trainable_variables = [self.variance]
        

    def get_marginal_distribution(self, x):
        return tfp.distributions.Normal(loc=x, scale=self.variance)


@pytest.fixture
def dummy_model():
    return DummyModel()


def test_nllbase_initialization(dummy_model):
    nll = NLLBase(dummy_model, feature_ndims=2, dtype="float64")
    assert nll.model == dummy_model
    assert nll.feature_ndims == 2
    assert nll.dtype == "float64"
    assert isinstance(nll.optimizer, tf.optimizers.Adam)


@pytest.mark.parametrize(
    "dummy_model,feature_ndims,dtype",
    [
        (None, 2, "float32"),
        (dummy_model, -1, "float64"),
        (dummy_model, 2, "int32"),
    ],
)
def test_nllbase_sc(dummy_model, feature_ndims, dtype):
    with pytest.raises(AssertionError):
        NLLBase(dummy_model, feature_ndims, dtype)

def test_nllbase_loss(dummy_model, ):
    nll = NLLBase(dummy_model, 1, dtype = "float32")
    x = np.array([1.0, 2.0, 3.0], dtype = "float32")
    y = np.array([1.0, 2.0, 3.0], dtype = "float32")
    loss = nll.loss(x, y)
    
    assert isinstance(loss, tf.Tensor)
    assert (tf.math.is_nan(loss) == False).numpy().all()
    assert (tf.math.is_inf(loss) == False).numpy().all()


def test_nllbase_optimize(dummy_model):
    nll = NLLBase(dummy_model, 1, dtype = "float32")
    x = np.array([1.0, 2.0, 3.0], dtype = "float32")
    y = np.array([1.0, 2.0, 3.0], dtype = "float32")
    loss = nll.optimize(x, y)
    assert isinstance(loss, tf.Tensor)
    assert (tf.math.is_nan(loss) == False).numpy().all()
    assert (tf.math.is_inf(loss) == False).numpy().all()
    assert (tf.math.is_nan(nll.model.variance) == False).numpy().all()
    assert (tf.math.is_inf(nll.model.variance) == False).numpy().all()  


def test_nllbase_fit(dummy_model):
    nll = NLLBase(dummy_model)
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    nll_value = nll.fit(x, y, n_restart=5, verbose=False)
    assert (tf.math.is_nan(nll_value) == False).numpy().all()
    assert (tf.math.is_inf(nll_value) == False).numpy().all()
    assert (tf.math.is_nan(nll.model.variance) == False).numpy().all()
    assert (tf.math.is_inf(nll.model.variance) == False).numpy().all()  

@pytest.mark.parametrize(
    "x,y",
    [
        (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0])),
        ([1.0, 2.0, 3.0], np.array([1.0, 2.0, 3.0])),
    ],
)
def test_nllbase_loss_invalid_input(dummy_model, x,y):
    nll = NLLBase(dummy_model)
    with pytest.raises(AssertionError):
        nll.loss(x, y)

    with pytest.raises(AssertionError):
        nll.optimize(x, y)

    with pytest.raises(AssertionError):
        nll.fit(x, y)
