import numpy as np
import pytest
from networkx import DiGraph
from data_struct import hDict, GraphObj, Var, Node
from surrogates.prior.base import PriorBase
from collections import OrderedDict

@pytest.fixture
def graph_obj():
    G = DiGraph()
    G.add_edges_from(
        [
            ("A_0", "B_0"),
            ("A_0", "C_0"),
            ("B_1", "B_2"),
            ("C_1", "C_2"),
            ("A_1", "A_2"),
            ("A_1", "B_2"),
        ]
    )
    target_var = Var(
        name="C",
    )
    return GraphObj(G, nT=3, target_var=target_var)


@pytest.fixture
def prior_base(graph_obj):
    return PriorBase(graph_obj)


def test_priorbase_initialization(prior_base, graph_obj):
    assert prior_base.G == graph_obj
    assert prior_base.nT == graph_obj.nT
    assert prior_base.nVar == graph_obj.nVar

    for node1, node2 in zip(prior_base.nodes, graph_obj.nodes):
        assert node1.name == node2.name
        assert node1.t == node2.t
    assert prior_base.dtype == "float32"
    assert isinstance(prior_base.f, hDict)
    assert prior_base.f.nT == graph_obj.nT
    assert prior_base.f.nTrials == 1


def test_priorbase_get_M(prior_base):
    expected_matrix = np.array(
        [
            # A_0, B_0, C_0, B_1, C_1, A_1, C_2, A_2, B_2
            [0, 1, 1, 0, 0, 0, 0, 0, 0],  # A_0
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # B_0
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # C_0
            [0, 0, 0, 0, 0, 0, 0, 0, 1],  # B_1
            [0, 0, 0, 0, 0, 0, 1, 0, 0],  # C_1
            [0, 0, 0, 0, 0, 0, 0, 1, 1],  # A_1
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # C_2
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # A_2
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # B_2
        ]
    )
    result_matrix = prior_base.get_M()
    assert np.array_equal(result_matrix, expected_matrix)


def test_priorbase_fork_node(prior_base):
    data = hDict(
        variables = prior_base.G.variables,
        nT = 3,
        nTrials = 1,
    )
    def fork_ops(pa_node, pa_value, ch_node, ch_value, i, funcs):
        prior_base.f.add((pa_node.name, i, ch_node.name))
        return {(pa_node, ch_node): "fork"}
    
    prior_base.fork_ops = fork_ops

    A = prior_base.get_M()
    A, funcs = prior_base.fork_node(A, data)

    expected_matrix = np.array(
        [
            # A_0, B_0, C_0, B_1, C_1, A_1, C_2, A_2, B_2
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # A_0
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # B_0
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # C_0
            [0, 0, 0, 0, 0, 0, 0, 0, 1],  # B_1
            [0, 0, 0, 0, 0, 0, 1, 0, 0],  # C_1
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # A_1
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # C_2
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # A_2
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # B_2
        ]
    )
    assert np.array_equal(A, expected_matrix)
    assert funcs == OrderedDict([((Node('A', 0), Node('B', 0)), 'fork'), ((Node('A', 0), Node('C', 0)), 'fork'), ((Node('A', 1), Node('A', 2)), 'fork'), ((Node('A', 1), Node('B', 2)), 'fork')])


def test_priorbase_normal_node(prior_base):
    data = hDict(
        variables=prior_base.G.variables,
        nT=3,
        nTrials=1,
    )

    def normal_ops(pa_node, pa_value, ch_node, ch_value, funcs):
        prior_base.f.add((pa_node.name,))
        return {(pa_node, ch_node): "normal"}

    prior_base.normal_ops = normal_ops

    A = prior_base.get_M()
    A, funcs = prior_base.normal_node(A, data)

    expected_matrix = np.array(
        [
            # A_0, B_0, C_0, B_1, C_1, A_1, C_2, A_2, B_2
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # A_0
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # B_0
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # C_0
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # B_1
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # C_1
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # A_1
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # C_2
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # A_2
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # B_2
        ]
    )
    assert np.array_equal(A, expected_matrix)
    assert funcs == OrderedDict([((Node('A', 0), Node('B', 0)), "normal"), ((Node('A', 0), Node('C', 0)), "normal"), ((Node('B', 1), Node('B', 2)), "normal"), ((Node('C', 1), Node('C', 2)), "normal"), ((Node('A', 1), Node('A', 2)), "normal"), ((Node('A', 1), Node('B', 2)), "normal")])


def test_priorbase_collider_node(prior_base):
    data = hDict(
        variables=prior_base.G.variables,
        nT=3,
        nTrials=1,
    )

    def collider_ops(pa_nodes, pa_values, ch_node, ch_value, funcs):
        prior_base.f.add(tuple(pa_node.name for pa_node in pa_nodes))
        return {tuple(pa_node for pa_node in pa_nodes): "collider"}

    prior_base.collider_ops = collider_ops

    A = prior_base.get_M()
    A, funcs = prior_base.collider_node(A, data)
    expected_matrix = np.array(
        [
            # A_0, B_0, C_0, B_1, C_1, A_1, C_2, A_2, B_2
            [0, 1, 1, 0, 0, 0, 0, 0, 0],  # A_0
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # B_0
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # C_0
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # B_1
            [0, 0, 0, 0, 0, 0, 1, 0, 0],  # C_1
            [0, 0, 0, 0, 0, 0, 0, 1, 0],  # A_1
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # C_2
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # A_2
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # B_2
        ]
    )
    assert np.array_equal(A, expected_matrix)
    assert funcs == OrderedDict([((Node('A',1), Node('B',1)), 'collider')]) or funcs == OrderedDict([((Node('B',1), Node('A',1)), 'collider')])

