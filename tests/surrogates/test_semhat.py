import pytest
import numpy as np
from typing import List, Dict, Any
from unittest.mock import MagicMock
from data_struct import GraphObj, hDict, Node, Var
from surrogates  import PriorEmit, PriorTrans, SEMHat

# Fixtures for common test data

@pytest.fixture
def mock_graph_obj():
    class MockDAG:
        def __init__(self):
            self.predecessors = {
                "X_1": ["X_0"],
                "Y_1": ["Y_0", "X_1"],
                "X_0": [],
                "Y_0": [],
            }
            self.in_degree = {
                "X_0": 0,
                "X_1": 1,
                "Y_0": 0,
                "Y_1": 2,
            }
    
    class MockGraph(GraphObj):
        def __init__(self):
            self.nVar = 2
            self.nT = 2
            self.variables = ["X", "Y"]
            self.nodes = [
                Node("X", 0),
                Node("X", 1),
                Node("Y", 0),
                Node("Y", 1),
            ]
            self.dag = MockDAG()
    
    return MockGraph()

@pytest.fixture
def mock_hdict_data():
    data = hDict(variables = [Var('X'), Var('Y')], default = lambda x,y : np.random.randn(x,y).astype("float32"))
    return data

@pytest.fixture
def sem_hat(mock_graph_obj, mock_hdict_data):
    return SEMHat(mock_graph_obj, mock_hdict_data, dtype="float32")

# Test cases

# def test_init_valid(sem_hat, mock_graph_obj, mock_hdict_data):
#     assert sem_hat.G == mock_graph_obj
#     assert sem_hat.nVar == 2
#     assert sem_hat.nT == 2
#     assert sem_hat.dtype == "float32"
#     assert isinstance(sem_hat.gp_emit, PriorEmit)
#     assert isinstance(sem_hat.gp_trans, PriorTrans)

# def test_init_invalid_graph_type(mock_hdict_data):
#     with pytest.raises(AssertionError) as e:
#         SEMHat("invalid_graph", mock_hdict_data)
#     assert "Expected GraphObj" in str(e.value)

# def test_init_data_keys_mismatch(mock_graph_obj):
#     invalid_data = hDict()
#     invalid_data["X"] = np.random.randn(2, 100)
#     with pytest.raises(AssertionError) as e:
#         SEMHat(mock_graph_obj, invalid_data)
#     assert "Data keys must match" in str(e.value)

# def test_filter_pa_t_valid(sem_hat):
#     parents = sem_hat.filter_pa_t(Node("X", 1), 0)
#     assert len(parents) == 1
#     assert parents[0].gstr == "X_0"

# def test_filter_pa_t_invalid_time(sem_hat):
#     with pytest.raises(AssertionError):
#         sem_hat.filter_pa_t("X_1", -1)
#     with pytest.raises(AssertionError):
#         sem_hat.filter_pa_t("X_1", 2)

# def test_get_edgekeys_transition(mocker, sem_hat):
#     mocker.patch.object(sem_hat.gp_trans.f["X"], "__getitem__", return_value=MagicMock())
#     node = Node("X", 1)
#     edge_keys = sem_hat.get_edgekeys(node, 0)
#     assert edge_keys == ("X_0", 0, 1)

# def test_get_edgekeys_emission(mocker, sem_hat):
#     mocker.patch.object(sem_hat.gp_emit.f["Y"], "__getitem__", return_value=MagicMock())
#     node = Node("Y", 1)
#     edge_keys = sem_hat.get_edgekeys(node, 1)
#     assert edge_keys == ("X_1", 1, 1)

# def test_update_prior(sem_hat, mock_hdict_data):
#     sem_hat.gp_emit.fit = MagicMock()
#     sem_hat.gp_trans.fit = MagicMock()
#     sem_hat.update_prior(mock_hdict_data)
#     sem_hat.gp_emit.fit.assert_called_once_with(mock_hdict_data)
#     sem_hat.gp_trans.fit.assert_called_once_with(mock_hdict_data)

# def test_select_sample_fork(sem_hat, mock_hdict_data):
#     edge_key = ("X_0", 0, 1)
#     sample = sem_hat.select_sample(mock_hdict_data, edge_key, 0, 10)
#     assert sample.shape == (10, -1)

# def test_get_gp_emit_mean(sem_hat, mocker):
#     mock_predict = mocker.MagicMock(return_value=(np.zeros(10), np.ones(10)))
#     mocker.patch.object(sem_hat.gp_emit.f["X"][0, 0], "predict", mock_predict)
#     gp_func = sem_hat.get_gp_emit(0)
#     result = gp_func(0, [], ["X"], {}, 10)
#     assert result.shape == (10,)

# def test_static_function(sem_hat):
#     static_func = sem_hat.static(0)
#     assert isinstance(static_func, hDict)
#     assert "X" in static_func
#     assert static_func["X"][0, 0] is not None

# def test_dynamic_function(sem_hat):
#     dynamic_func = sem_hat.dynamic(0)
#     assert isinstance(dynamic_func, hDict)
#     assert "X" in dynamic_func
#     assert dynamic_func["X"][0, 0] is not None