import pytest
import numpy as np
import tensorflow as tf
from networkx import DiGraph

from data_struct import hDict, Node, Var, GraphObj
from models import GPRegression, KernelDensity
from surrogates.prior.emit import PriorEmit

@pytest.fixture
def setup_prior_emit():
    # G = None  # Replace with appropriate graph structure
    G = DiGraph()
    G.add_edges_from(
        [
            ("A_0", "B_0"),
            ("A_0", "C_0"),
            ("B_1", "B_2"),
            ("A_1", "B_1"),
            ("C_0", "C_1"),
            ("C_1", "C_2"),
            ("A_1", "A_2"),
            ("C_1", "B_1"),
        ]
    )

    G = GraphObj(G, nT=3, target_var=Var(name="C"))
    return PriorEmit(G)

def test_fork_ops(setup_prior_emit):
    prior_emit = setup_prior_emit
    t = 0
    pa_node = Node(name="A", t=t)
    ch_node = Node(name="B", t=t)
    pa_value = np.array([1.0, 2.0, 3.0])
    ch_value = np.array([4.0, 5.0, 6.0])
    funcs = hDict(variables=[], nT=3, nTrials=1)

    result = prior_emit.fork_ops(pa_node, pa_value, ch_node, ch_value, 0, funcs)
    assert isinstance(result, hDict)
    assert isinstance(result[(pa_node.name, 0, ch_node.name)][t, 0], GPRegression)

    ch_node2 = Node(name="C", t=t)
    ch_value2 = np.array([6.0, 5.0, 4.0])

    result = prior_emit.fork_ops(pa_node, pa_value, ch_node2, ch_value2, 1, funcs)
    assert isinstance(result, hDict)
    assert isinstance(result[(pa_node.name, 0, ch_node.name)][t, 0], GPRegression)
    assert isinstance(result[(pa_node.name, 1, ch_node2.name)][t, 0], GPRegression)


def test_source_ops(setup_prior_emit):
    prior_emit = setup_prior_emit
    t = 0
    pa_node = Node(name="A", t=t)
    pa_value = np.array([1.0, 2.0, 3.0])
    funcs = hDict(variables=[], nT=3, nTrials=1)
    
    result = prior_emit.source_ops(pa_node, pa_value, funcs)
    assert isinstance(result, hDict)
    assert isinstance(result[(None, pa_node.name)][t, 0], KernelDensity)

    t = 1
    pa_node = Node(name="B", t=t)
    pa_value = np.array([1.0, 2.0, 3.0])
    
    result = prior_emit.source_ops(pa_node, pa_value, funcs)
    assert isinstance(result, hDict)
    assert isinstance(result[(None, Var('A'))][0, 0], KernelDensity)
    assert isinstance(result[(None, pa_node.name)][t, 0], KernelDensity)


def test_normal_ops_invalid(setup_prior_emit):
    prior_emit = setup_prior_emit

    pa_node = Node(name="A", t=1)
    ch_node = Node(name="B", t=2)
    pa_value = np.array([1.0, 2.0, 3.0])
    ch_value = np.array([4.0, 5.0, 6.0])
    funcs = hDict(variables=[], nT=3, nTrials=1)
    
    result = prior_emit.normal_ops(pa_node, pa_value, ch_node, ch_value, funcs)
    assert isinstance(result, hDict)
    assert result.get((pa_node.name,)) is None

def test_normal_ops(setup_prior_emit):
    prior_emit = setup_prior_emit
    t = 1
    pa_node = Node(name="A", t=t)
    ch_node = Node(name="B", t=t)
    pa_value = np.array([1.0, 2.0, 3.0])
    ch_value = np.array([4.0, 5.0, 6.0])
    funcs = hDict(variables=[], nT=3, nTrials=1)

    result = prior_emit.normal_ops(pa_node, pa_value, ch_node, ch_value, funcs)
    assert isinstance(result, hDict)
    assert isinstance(result[(pa_node.name, )][t, 0], GPRegression)

    t = 2
    pa_node2 = Node(name="B", t=t)
    pa_value2 = np.array([3.0, 2.0, 1.0])
    ch_node2 = Node(name="C", t=t)
    ch_value2 = np.array([6.0, 5.0, 4.0])

    result = prior_emit.normal_ops(pa_node2, pa_value2, ch_node2, ch_value2, funcs)
    assert isinstance(result, hDict)
    assert isinstance(result[(pa_node2.name, )][2, 0], GPRegression)
    assert isinstance(result[(pa_node.name, )][1, 0], GPRegression)
    assert result[(pa_node.name, )][2, 0] is None
    
def test_collider_ops(setup_prior_emit):
    prior_emit = setup_prior_emit
    pa_nodes = [Node(name="parent1", t=0), Node(name="parent2", t=0)]
    pa_values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    ch_node = Node(name="child", t=0)
    ch_value = np.array([7.0, 8.0, 9.0])
    funcs = hDict(variables=[], nT=3, nTrials=1)
    
    result = prior_emit.collider_ops(pa_nodes, pa_values, ch_node, ch_value, funcs)
    assert isinstance(result, hDict)
    assert isinstance(result[tuple(pa_node.name for pa_node in pa_nodes) ][0,0] , GPRegression)
    
    
    pa_nodes2 = np.array([Node(name="parent3", t=1), Node(name="parent4", t=1),  Node(name="parent5", t=1), ])
    pa_values2 = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]])
    ch_node2 = Node(name="child", t=1)
    ch_value2 = np.array([7.0, 8.0, 9.0])

    result = prior_emit.collider_ops(pa_nodes2, pa_values2, ch_node2, ch_value2, funcs)
    assert isinstance(result, hDict)
    assert isinstance(result[tuple(pa_node.name for pa_node in pa_nodes) ][0,0] , GPRegression)
    assert isinstance(result[tuple(pa_node.name for pa_node in pa_nodes2) ][1,0] , GPRegression)

def test_source_node(setup_prior_emit):
    prior_emit = setup_prior_emit
    data = hDict(variables=prior_emit.variables, nT=3, nTrials=1, default = lambda x, y : np.random.rand(x, y))
    M, _ = prior_emit.get_M()

    result_A, result_funcs = prior_emit.source_node(M, data)
        
    source_nodes = [Node(name="A", t=0), Node(name="A", t=1), ]
    
    for var in [Var('A'), Var('B'), Var('C')]:
        
        for t in range(3):
            if Node(var, t) in source_nodes:
                assert isinstance(result_funcs.get((None, var.name))[t,0] , KernelDensity)
            else:
                if result_funcs.get((None, var.name)) is not None:
                    assert result_funcs.get((None, var.name))[t,0] is None

    assert np.array_equal(result_A, M)
    assert isinstance(result_funcs, hDict)

def test_get_M(setup_prior_emit):
    prior_emit = setup_prior_emit
    _, result = prior_emit.get_M()
    
    # [A_0 B_0 C_0 A_1 B_1 C_1 A_2 B_2 C_2]
    expected = np.array(
        [[0., 1., 1., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.],]
    )
    
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, expected)

def test_fit(setup_prior_emit):
    prior_emit = setup_prior_emit
    data = hDict(variables=prior_emit.variables, nT=3, nTrials=1, default=lambda x, y: np.random.rand(x, y))
    haha = prior_emit.fit(data)
    
    arr = [
        ((Var('A'), 0, Var('B')), [[GPRegression], [None], [None]]), # A_0 -> B_0
        ((Var('A'), 1, Var('C')), [[GPRegression], [None], [None]]), # A_0 -> C_0
        ((None, Var('A')), [[KernelDensity], [KernelDensity], [None]]), # A_0, A_1, 
        ((Var('A'),), [[None], [GPRegression], [None]]), # A_1 -> B_1
        ((Var('C'),), [[None], [GPRegression], [None]]), # C_1 -> B_1
        ((Var('A'), Var('C')), [[None], [GPRegression], [None]]), # A_1 -> C_1, B_1 -> C_1
    ]
    '''
    all edges
        ("A_0", "B_0"),
        ("A_0", "C_0"),
        ("B_1", "B_2"),
        ("A_1", "B_1"),
        ("C_0", "C_1"),
        ("C_1", "C_2"),
        ("A_1", "A_2"),
        ("C_1", "B_1"),
    source nodes: A_0, A_1, 
    emission edges
        ("A_0", "B_0"),
        ("A_0", "C_0"),
        ("A_1", "B_1"),
        ("C_1", "B_1"),

    '''
    for item in arr:
        key, value = item
        for t in range(3):
            if value[t][0] is not None:
                assert isinstance(haha[key][t][0], value[t][0]) 
            else:
                assert haha[key][t][0] is None