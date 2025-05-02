import pytest
import numpy as np
import tensorflow as tf
from models.glm import GLMTanh

def test_glmtanh_forward():
    model = GLMTanh(A=2.0, B=0.5, trainable=False)
    x = tf.constant([[1.0], [2.0]], dtype=tf.float32)
    y = model(x)
    expected = 2.0 * tf.math.tanh(0.5 * x)
    np.testing.assert_allclose(y.numpy(), expected.numpy(), rtol=1e-5)

def test_glmtanh_trainable_variables():
    model = GLMTanh(trainable=True)
    assert len(model.trainable_variables) == 2
    assert all(isinstance(v, tf.Variable) for v in model.trainable_variables)

def test_glmtanh_not_trainable():
    model = GLMTanh(trainable=False)
    assert model.trainable_variables == []

def test_glmtanh_optimize_reduces_loss():
    # Simple data: y = tanh(x)
    x = tf.constant(np.linspace(-2, 2, 100).reshape(-1, 1), dtype=tf.float32)
    y = tf.math.tanh(x)
    model = GLMTanh(A=1.0, B=2.0, trainable=True)
    initial_loss = model.optimize(x, y).numpy()
    for _ in range(10):
        loss = model.optimize(x, y).numpy()
    assert loss < initial_loss

def test_glmtanh_fit_verbose(capsys):
    x = tf.constant(np.linspace(-1, 1, 10).reshape(-1, 1), dtype=tf.float32)
    y = 0.5 * tf.math.tanh(2.0 * x)
    model = GLMTanh(trainable=True)
    model.fit(x, y, n_restart=10, verbose=True)
    captured = capsys.readouterr()
    assert "Final mse" in captured.out
