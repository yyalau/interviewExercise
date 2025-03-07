import pytest
import numpy as np
from data_ops.sampler.observation import DSamplerObsBase, DSamplerObsBF, DSamplerObsDCBO
from sems import SEMBase
from data_struct import Var, hDict
from collections import OrderedDict

class MockSEM(SEMBase):
    @staticmethod
    def static() -> OrderedDict:
        X = lambda noise, t, sample: noise
        Y = lambda noise, t, sample: sample["X"][t] + noise
        return OrderedDict([("X", X), ("Y", Y)])

    @staticmethod
    def dynamic() -> OrderedDict:
        X = lambda noise, t, sample: sample["X"][t - 1] + noise
        Y = lambda noise, t, sample: sample["X"][t] + sample["Y"][t - 1] + noise
        return OrderedDict([("X", X), ("Y", Y)])


@pytest.mark.parametrize(
    "sem, nT, variables, ", [
        (None, 1, [Var("X")], ),
        (MockSEM(), -1, [Var("X")],  ),
        (MockSEM(), 1, Var("X"), ),
        (MockSEM(), 1, ["123"],  ),
    ]
)
def test_DSamplerObsBase_init_invalid(sem, nT, variables,):
    with pytest.raises(AssertionError):
        DSamplerObsBase(sem, nT, variables)


@pytest.mark.parametrize(
    "initial_values, interv_levels, epsilon, n_samples, expected_exception, match", [
        ("invalid", None, None, 1, AssertionError, "initial_values must be None or a dictionary"),
        (None, "invalid", None, 1, AssertionError, "interv_levels must be None or a dictionary"),
        (None, None, "invalid", 1, AssertionError, "epsilon must be None or a dictionary"),
        (None, None, None, -1, AssertionError, "n_samples must be a positive integer"),
    ]
)
def test_DSamplerObsBase_sample_invalid(initial_values, interv_levels, epsilon, n_samples, expected_exception, match):
    sem = MockSEM()
    sampler = DSamplerObsBase(sem, 1, ["X", "Y"], "float32")
    with pytest.raises(expected_exception, match=match):
        sampler.sample(initial_values, interv_levels, epsilon, n_samples)

def test_DSamplerObsBase_sample_valid():
    sem = MockSEM()
    sampler = DSamplerObsBase(sem, 1, ["X", "Y"], "float32")
    n_samples = 2
    samples = sampler.sample(n_samples=n_samples)
    
    assert isinstance(samples, hDict)
    assert "X" in samples
    assert "Y" in samples
    assert samples["X"].shape == (1, n_samples)
    assert samples["Y"].shape == (1, n_samples)
    
    # Check the values in the samples
    epsilon = np.random.randn(1, n_samples)
    expected_X = epsilon
    expected_Y = expected_X + epsilon
    
    np.testing.assert_array_almost_equal(samples["X"][0, :], expected_X[0, :])
    np.testing.assert_array_almost_equal(samples["Y"][0, :], expected_Y[0, :])

def test_DSamplerObsBase_sample_dynamic():
    sem = MockSEM()
    sampler = DSamplerObsBase(sem, 2, ["X", "Y"], "float32")
    n_samples = 2
    samples = sampler.sample(n_samples=n_samples)
    
    assert isinstance(samples, hDict)
    assert "X" in samples
    assert "Y" in samples
    assert samples["X"].shape == (2, n_samples)
    assert samples["Y"].shape == (2, n_samples)
    
    # Check the values in the samples
    epsilon = np.random.randn(2, n_samples)
    expected_X = np.zeros((2, n_samples))
    expected_Y = np.zeros((2, n_samples))
    
    expected_X[0, :] = epsilon[0, :]
    expected_Y[0, :] = expected_X[0, :] + epsilon[0, :]
    
    expected_X[1, :] = expected_X[0, :] + epsilon[1, :]
    expected_Y[1, :] = expected_X[1, :] + expected_Y[0, :] + epsilon[1, :]
    
    np.testing.assert_array_almost_equal(samples["X"][0, :], expected_X[0, :])
    np.testing.assert_array_almost_equal(samples["Y"][0, :], expected_Y[0, :])
    np.testing.assert_array_almost_equal(samples["X"][1, :], expected_X[1, :])
    np.testing.assert_array_almost_equal(samples["Y"][1, :], expected_Y[1, :])