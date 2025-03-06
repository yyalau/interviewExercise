from data_ops.base import DatasetBase, DataSamplerBase
from sems import SEMBase
import pytest

@pytest.mark.parametrize(
    "nT,n_samples", [
        (-1, 0),
        (0, 1)
    ]
)
def test_DatasetBase(nT, n_samples, ):
    variables = [1,2,3]
    
    with pytest.raises(AssertionError):
        DatasetBase(nT, n_samples, variables, dtype="float32")  


@pytest.mark.parametrize(
    "sem,nT", [
        (SEMBase, 0),
        (1, -1)
    ]
)
def test_DataSamplerBase(sem, nT):
    variables = [1,2,3]
    with pytest.raises(AssertionError):
        DataSamplerBase(sem, nT, variables, dtype="float32") 