import pytest
import numpy as np
from typing import List, Dict, Any
from unittest.mock import MagicMock
from data_struct import GraphObj, hDict, Node, Var
from surrogates import PriorEmit, PriorTrans, SEMHat
from networkx import DiGraph
from utils.tools import tnode2var
import tensorflow as tf
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
        nT=3,
        default=lambda x, y: np.random.randn(x, y).astype("float32"),
    )
    return data


@pytest.fixture
def sem_hat(mock_graph_obj, mock_hdict_data):
    return SEMHat(mock_graph_obj, mock_hdict_data, dtype="float32")


# Test cases


def test_init_valid(
    sem_hat,
):
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
    ],
)
def test_init_data_keys_mismatch(mock_graph_obj, invalid_data):
    with pytest.raises(AssertionError) as e:
        SEMHat(mock_graph_obj, invalid_data)
    assert "Data keys must match" in str(e.value)

@pytest.mark.parametrize(
    "node_s, t, result",
    [
        ("C_2", 1, {Node("C", 1)}),
        ("B_2", 1, {Node("A", 1), Node("B", 1), Node("C", 1)}),
        ("B_2", 2, {Node("A", 2)}),
    ]   
)
def test_filter_pa_t_valid(sem_hat, node_s, t, result):
    parents = sem_hat.filter_pa_t(node_s, t)
    assert set(parents) == result


@pytest.mark.parametrize(
    "node, time",
    [
        ("A_1", 2),
        ("A_2", 0),
        ("B_3", 2),
    ],
)
def test_filter_pa_t_invalid_time(sem_hat, node, time):
    with pytest.raises(AssertionError):
        sem_hat.filter_pa_t(node, time)


def test_filter_pa_t_invalid_node(sem_hat):
    with pytest.raises(AssertionError):
        sem_hat.filter_pa_t(Node("D", 1), 2)


"""
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
"""
@pytest.mark.parametrize(
    "node, time, expected",
    [
        (Node("A", 1), 0, ()),
        (Node("A", 2), 1, (Node("A", 1), 0, Node("A", 2))),
        (Node("B", 2), 1, (Node("A", 1), Node("B", 1), Node("C", 1))),
        (Node("B", 2), 2, (Node("A", 2),)),
        (Node("C", 1), 0, (Node('A',0), Node("C", 0),)),
    ],
)
def test_get_edgekeys(sem_hat, node, time, expected):

    edge_keys = sem_hat.get_edgekeys(node, time)
    print(sem_hat.gp_trans.f.keys())
    assert edge_keys == expected


def test_update_prior(sem_hat, mock_hdict_data):
    sem_hat.gp_emit.fit = MagicMock()
    sem_hat.gp_trans.fit = MagicMock()
    sem_hat.update_prior(mock_hdict_data)
    sem_hat.gp_emit.fit.assert_called_once_with(mock_hdict_data)
    sem_hat.gp_trans.fit.assert_called_once_with(mock_hdict_data)

@pytest.mark.parametrize(
    "edge_key, n_samples",
    [
        ((Node("A", 1), 0, Node("A", 2)), 10, ),
        ((Node("A", 1), Node("B", 1), Node("C", 1)), 20),
        ((Node("A", 0),), 3),
    ]
)
def test_select_sample(sem_hat, edge_key, n_samples):
    sample_data = hDict(
        variables=[Var("A"), Var("B"), Var("C")],
        nT=3,
        nTrials = n_samples,
        default=lambda x, y: np.random.randn(x, y).astype("float32"),
    )
    sample = sem_hat.select_sample(sample_data, edge_key, n_samples)
    assert sample.shape == (n_samples, 1)


@pytest.mark.parametrize(
    "edge_key",
    [
        (None, Node("A", 0),),
        (None, Node("A", 1),),
    ]
)
def test_get_kernel(sem_hat, edge_key):
    func = sem_hat.get_kernel()
    result = func(edge_key, 10)
    assert result.shape == (10,)

@pytest.mark.parametrize(
    "emit_keys, trans_keys, edge_type",
    [
        (None, (Node("A", 0), 0, Node("B", 1)), "trans"),
        (None, (Node("B", 1),), "trans"),
        (None, (Node("A", 1),Node("B", 1),Node("C", 1),),"trans"),
        ((Node("A", 2),), None, "emit"),
        ((Node("C", 0), ), (Node("C", 0),), "both"),
        ((Node("C", 0), ), None, "emit"),

    ]
)
def test_get_gp_callable(sem_hat, emit_keys, trans_keys, edge_type):
    n_samples = 10
    moment = 0
    gp_func = sem_hat.get_gp_callable(moment)
    
    sample_data = hDict(
        variables=[Var("A"), Var("B"), Var("C")],
        nT=3,
        nTrials = n_samples,
        default=lambda x, y: np.random.randn(x, y).astype("float32"),
    )
    
    result = gp_func(trans_keys, emit_keys, sample_data, n_samples)
    
    if edge_type == "emit":
    
        expected = tf.reshape(
            sem_hat.gp_emit.f[tnode2var(emit_keys)][emit_keys[0].t, 0].predict(
            sem_hat.select_sample(sample_data, emit_keys, n_samples)
        )[moment], (-1,),)
    
    elif edge_type == 'trans':
        expected = tf.reshape(
            sem_hat.gp_trans.f[tnode2var(trans_keys)][trans_keys[0].t, 0].predict(
            sem_hat.select_sample(sample_data, trans_keys, n_samples)
        )[moment], (-1,),)
        
    elif edge_type == 'both':
        expected = tf.reshape(
            sem_hat.gp_emit.f[tnode2var(emit_keys)][emit_keys[0].t, 0].predict(
            sem_hat.select_sample(sample_data, emit_keys, n_samples)
        )[moment], (-1,),) + tf.reshape(
            sem_hat.gp_emit.f[tnode2var(trans_keys)][trans_keys[0].t, 0].predict(
            sem_hat.select_sample(sample_data, trans_keys, n_samples)
        )[moment], (-1,),) 
        
    assert np.array_equal(result, expected)


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
