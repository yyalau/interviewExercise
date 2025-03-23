import pytest
import numpy as np
from typing import List, Dict, Any
from unittest.mock import MagicMock
from data_struct import GraphObj, hDict, Node, Var
from surrogates import PriorEmit, PriorTrans, SEMHat
from networkx import DiGraph

# Fixtures for common test data


@pytest.fixture
def mock_graph_obj():
    G = DiGraph()
    G.add_edges_from(
        [
            ("A_0", "B_1"),
            ("A_0", "C_1"),
            ("B_1", "B_2"),
            ("A_1", "B_2"),
            ("C_0", "C_1"),
            ("C_1", "C_2"),
            ("A_1", "A_2"),
            ("C_1", "B_2"),
            ("C_0", "B_0"),
            ("A_2", "B_2"),
        ]
    )

    G = GraphObj(G, nT=3, target_var=Var(name="C"))
    return G

@pytest.fixture
def mock_hdict_data():
    data = hDict(
        variables=[Var("A"), Var("B"), Var("C")],
        nT = 3,
        default=lambda x, y: np.random.randn(x, y).astype("float32"),
    )
    return data


@pytest.fixture
def sem_hat(mock_graph_obj, mock_hdict_data):
    return SEMHat(mock_graph_obj, mock_hdict_data, dtype="float32")


# Test cases

def test_init_valid(sem_hat, ):
    assert isinstance(sem_hat.G, GraphObj)
    assert sem_hat.nVar == 3
    assert sem_hat.nT == 3
    assert sem_hat.dtype == "float32"
    assert isinstance(sem_hat.gp_emit, PriorEmit)
    assert isinstance(sem_hat.gp_trans, PriorTrans)


@pytest.mark.parametrize(
    "invalid_data",
    [
        hDict(
            variables=[Var("A"), Var("B"), Var("C"), Var("D")],
            nT=3,
            default=lambda x, y: np.random.randn(x, y).astype("float32"),
        ),
        hDict(
            variables=[Var("A"), Var("B")],
            nT=3,
            default=lambda x, y: np.random.randn(x, y).astype("float32"),
        ),
    ]
)
def test_init_data_keys_mismatch(mock_graph_obj, invalid_data):
    with pytest.raises(AssertionError) as e:
        SEMHat(mock_graph_obj, invalid_data)
    assert "Data keys must match" in str(e.value)

def test_filter_pa_t_valid(sem_hat):
    parents = sem_hat.filter_pa_t(Node("C", 2), 1)
    assert set(parents) == {Node('C', 1), }    

    parents = sem_hat.filter_pa_t(Node("B", 2), 1)
    assert set(parents) == {Node('A', 1), Node('B', 1), Node('C', 1)}    

    parents = sem_hat.filter_pa_t(Node("B", 2), 2)
    assert set(parents) == {Node('A', 2), }


@pytest.mark.parametrize(
    "node, time",   
    [
        ("A_1", 0),
        ("A_2", 1),
        ("B_2",2),
    ]
)
def test_filter_pa_t_invalid_time(sem_hat, node, time):
    with pytest.raises(AssertionError):
        sem_hat.filter_pa_t(node, time)


def test_filter_pa_t_invalid_node(sem_hat):
    with pytest.raises(AssertionError):
        sem_hat.filter_pa_t(Node('D', 1), 2)
        
'''
("A_0", "B_1"),
("A_0", "C_1"),
("B_1", "B_2"),
("A_1", "B_2"),
("C_0", "C_1"),
("C_1", "C_2"),
("A_1", "A_2"),
("C_1", "B_2"),
("C_0", "B_0"),
("A_2", "B_2"),
'''

@pytest.mark.parametrize(
    "node, time, expected",
    [
        (Node("A", 1), 0, ()),
        (Node("A", 2), 1, (Node('A', 1), 0, Node('A', 2) )),
        (Node("B", 2), 1, (Node('A', 1), 1,  Node('B', 2))),
        (Node("B", 2), 2, (Node('A', 2), )),
    ]
)
def test_get_edgekeys(sem_hat, node, time, expected):
    
    edge_keys = sem_hat.get_edgekeys(node, time)
    print(edge_keys)
    assert edge_keys == expected    
    
    
    
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
