import pytest
import numpy as np
from data_ops import DatasetBF

@pytest.mark.parametrize(
    "dataX, dataY, dtype", [
        (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), "float32"),
        (None, None, "float32"),
    ]
)
def test_DatasetBF_init(dataX, dataY, dtype):
    dataset = DatasetBF(dataX, dataY, dtype)
    
    if dataX is not None and dataY is not None:
        assert dataset.n_samples == dataX.shape[0]
    else:
        assert dataset.n_samples is None
        
        
@pytest.mark.parametrize(
    "dataX, dataY", [
        (np.array([[1, 2], [3, 4]]), np.array([[5, 6]]), ),
        (np.array([1, 2, 3]), np.array([[5, 6], [7, 8]]), ),
        (np.array([[1, 2], [3, 4]]), np.array([5, 6, 7, 8]), ),
        ("invalid", np.array([[5, 6], [7, 8]]), ),
        (np.array([[1, 2], [3, 4]]), None, ),
    ]
)
def test_DatasetBF_init_invalid(dataX, dataY, ):
    with pytest.raises(AssertionError):
        DatasetBF(dataX, dataY)

@pytest.mark.parametrize(
    "dataX, dataY, x, y, expected_dataX, expected_dataY", [
        (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), 
         np.array([[9, 10]]), np.array([[11, 12]]), 
         np.array([[1, 2], [3, 4], [9, 10]]), np.array([[5, 6], [7, 8], [11, 12]])),
        (None, None, 
         np.array([[1, 2]]), np.array([[3, 4]]), 
         np.array([[1, 2]]), np.array([[3, 4]])),
    ]
)
def test_DatasetBF_update(dataX, dataY, x, y, expected_dataX, expected_dataY):
    dataset = DatasetBF(dataX, dataY)
    dataset.update(x, y)
    
    assert np.array_equal(dataset.dataX, expected_dataX)
    assert np.array_equal(dataset.dataY, expected_dataY)


@pytest.mark.parametrize(
    "dataX", [np.array([[1, 2], [3, 4]]), None]
    )
@pytest.mark.parametrize(
    "dataY", [np.array([[5, 6], [7, 8]]), None]
)
@pytest.mark.parametrize(
    "x", [None, np.array([[9, 10, 11]])]
)
@pytest.mark.parametrize(
    "y", [None, np.array([[11, 12], [13,14]]),]
)
def test_DatasetBF_update_invalid(dataX, dataY, x, y):
    
    with pytest.raises(AssertionError):
        dataset = DatasetBF(dataX, dataY)
        dataset.update(x, y)