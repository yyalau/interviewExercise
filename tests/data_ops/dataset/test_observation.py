from data_ops import DatasetObsDCBO
import pytest
from data_struct import hDict
import numpy as np

@pytest.mark.parametrize(
    "nT", [1, 4,]    
)
@pytest.mark.parametrize(
    "n_old_data", [1, 3]    
)

@pytest.mark.parametrize(
    "n_new_data", [1, 5]    
)

def test_DatasetObsDCBO_update(nT, n_old_data, n_new_data):

    variables = [1, 2, 3]
    old = {}
    new = {}
    ans = {}
    params = ["initial_values", "interv_levels", "epsilon", "dataY"]

    for i, p in enumerate(params):
        old[p] = hDict(
            variables=variables,
            nT=nT if i !=0 else 1,
            nTrials=n_old_data if i !=0 and i!=1 else 1,
            default=lambda x, y: np.array([[i] * y] * x),
        )
        if i <=1: continue

        new[p] = hDict(
            variables=variables,
            nT=nT ,
            nTrials=n_new_data ,
            default=lambda x, y: np.array([[i + 4.0] * y] * x),
        )

        ans[p] = hDict(
            variables=variables,
            nT=nT,
            nTrials= n_old_data + n_new_data,
            default=lambda x, y: np.concatenate(
                (np.array([[i] * n_old_data] * nT), np.array([[i+4] * n_new_data] * nT)), axis = 1
            ) if i !=0 and i!=1 else np.array([[i + 4.0] * y] * x),
        )

    dataset = DatasetObsDCBO(**old)

    dataset.update( *[ new[p] for p in params[2:]])
    
    for p in params[2:]:
        for v in variables:
            if p == "dataY":
                assert np.allclose(dataset.dataY[v], ans[p][v], rtol = 0), f"dataset.dataY[{v}] = {dataset.dataY[v]}, ans[p][v] = {ans[p][v]}"
            else:
                assert np.allclose(dataset.dataX[p][v], ans[p][v], rtol = 0), f"dataset.dataX[{p}][{v}] = {dataset.dataX[p][v]}, ans[p][v] = {ans[p][v]}"

    assert dataset.n_samples == n_old_data + n_new_data, f"dataset.nTrials = {dataset.n_samples}, n_old_data + n_new_data = {n_old_data + n_new_data}"