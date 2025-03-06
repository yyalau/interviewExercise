import pytest
from data_struct import hDict
import numpy as np
from data_ops import DatasetInv
import tensorflow as tf


@pytest.mark.parametrize(
    "exp_sets", [[], [1,2], (1, 2), [(1,), (2,)]]
)
@pytest.mark.parametrize(
    "nT", [-1, "string"]
)
def test_DatasetInv_init(exp_sets , nT):
    nTrials = 3
    dtype = "float32"
        
    with pytest.raises(AssertionError):
        DatasetInv(exp_sets, nT, nTrials, dtype)

@pytest.fixture(
    params = [[(1,), (2,)], 2, 3, "float32"]
)
def test_DatasetInv_content(exp_sets, nT, nTrials, dtype):

    dataset = DatasetInv(exp_sets, nT, nTrials, dtype)
    for es in exp_sets:
        assert np.array_equal(dataset.n_samples[es], np.zeros((nT, 1)))
    
    assert set(dataset.dataX.keys()) == set(dataset.dataY.keys()) == set(exp_sets)
    for es in exp_sets:
        assert dataset.dataX[es].shape == (nT, nTrials, len(es))
        assert dataset.dataY[es].shape == (nT, nTrials)

@pytest.mark.parametrize(
    "es",  [(1,), (2,), (2,3)]
)
@pytest.mark.parametrize(
    "nT", [1, 2]
)
@pytest.mark.parametrize(
    "nTrials", [1, 3]
)
@pytest.mark.parametrize(
    "dtype", ["float32", ]
)
def test_DatasetInv_update(es, nT, nTrials, dtype):
    exp_sets =  [(1,), (2,), (2,3)]
    nT = 2
    nTrials = 3
    dtype = "float32"
    
    dataset = DatasetInv(exp_sets, nT, nTrials, dtype)
    
    t = 1
    x = np.array([-2.]*len(es), dtype=object)
    y = -1.
    
    dataset.update(es, t, x=x, y=y)
        
    assert np.array_equal(dataset.dataX[es][t, 0], x)
    assert dataset.dataY[es][t, 0] == y
    assert dataset.n_samples[es][t] == 1
    
    dataset.update(es, t, x=x, y=y)    
    assert np.array_equal(dataset.dataX[es][t, 1], x)
    assert dataset.dataY[es][t, 1] == y
    assert dataset.n_samples[es][t] == 2    



@pytest.mark.parametrize(
    "es, t, x, y", [
        ((3,), 0, np.array([-2.],), -1.),
        ((1,), -1, np.array([-2.], ), -1.),
        ((1,), 0, [-2.], -1.),
        ((2,3), 0, np.array([[-2.], ], ), -1.),
        ((1,), 0, np.array([-2.], ), "invalid"),
    ]
)

def test_update_invalid(es, t, x, y):
    exp_sets =  [(1,), (2,), (2,3)]
    nT = 2
    nTrials = 3
    dtype = "float32"
    
    dataset = DatasetInv(exp_sets, nT, nTrials, dtype)
    with pytest.raises(AssertionError):
        dataset.update(es, t, x=x, y=y)


@pytest.mark.parametrize(
    "es, t", [
        ((2,), 0),
        ((1,), 1),
        ((2,3), 2),
    ]
)

def test_DatasetInv_get(es, t):
    exp_sets =  [(1,), (2,), (2,3)]
    nT = 3
    nTrials = 3
    dtype = "float32"
    dataset = DatasetInv(exp_sets, nT, nTrials, dtype)
    
    # es = (1,)
    # t = 0
    x = np.array([-2.]*len(es), dtype=object)
    y = -1.
    
    dataset.update(es, t, x=x, y=y)
    
    x_out, y_out = dataset.get_tf(es, t)
    
    assert np.array_equal(x_out.numpy(), x.astype(dtype)[np.newaxis, ...])
    assert np.array_equal(y_out.numpy(), np.array([y], dtype=dtype))
