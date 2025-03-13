import pytest
import numpy as np
import tensorflow as tf
from networkx import DiGraph

from data_struct import hDict, Node, Var, GraphObj
from models import GPRegression
from surrogates.prior.trans import PriorTrans

@pytest.fixture
def setup_prior_trans():
    G = DiGraph()
    G.add_edges_from(
        [
            ("A_0", "B_1"),
            ("A_0", "C_1"),
            ("B_0", "B_2"),
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
    return PriorTrans(G)

def test_fork_ops(setup_prior_trans):
    prior_trans = setup_prior_trans
    t = 0
    pa_node = Node(name="A", t=t)
    ch_node = Node(name="B", t=t+1)
    pa_value = np.array([1.0, 2.0, 3.0])
    ch_value = np.array([4.0, 5.0, 6.0])
    funcs = hDict(variables=[], nT=3, nTrials=1)

    result = prior_trans.fork_ops(pa_node, pa_value, ch_node, ch_value, 0, funcs)
    assert isinstance(result, hDict)
    assert isinstance(result[(pa_node.name, 0, ch_node.name)][t, 0], GPRegression)

    ch_node2 = Node(name="C", t=t+1)
    ch_value2 = np.array([6.0, 5.0, 4.0])

    result = prior_trans.fork_ops(pa_node, pa_value, ch_node2, ch_value2, 1, funcs)
    assert isinstance(result, hDict)
    assert isinstance(result[(pa_node.name, 0, ch_node.name)][t, 0], GPRegression)
    assert isinstance(result[(pa_node.name, 1, ch_node2.name)][t, 0], GPRegression)


def test_normal_ops_invalid(setup_prior_trans):
    prior_trans = setup_prior_trans

    pa_node = Node(name="A", t=1)
    ch_node = Node(name="B", t=1)
    pa_value = np.array([1.0, 2.0, 3.0])
    ch_value = np.array([4.0, 5.0, 6.0])
    funcs = hDict(variables=[], nT=3, nTrials=1)
    
    result = prior_trans.normal_ops(pa_node, pa_value, ch_node, ch_value, funcs)
    assert isinstance(result, hDict)
    assert result.get((pa_node.name,)) is None

def test_normal_ops(setup_prior_trans):
    prior_trans = setup_prior_trans
    t = 1
    pa_node = Node(name="A", t=t)
    ch_node = Node(name="A", t=t+1)
    pa_value = np.array([1.0, 2.0, 3.0])
    ch_value = np.array([4.0, 5.0, 6.0])
    funcs = hDict(variables=[], nT=3, nTrials=1)

    result = prior_trans.normal_ops(pa_node, pa_value, ch_node, ch_value, funcs)
    assert isinstance(result, hDict)
    assert isinstance(result[(pa_node.name, )][t, 0], GPRegression)

    t = 1
    pa_node2 = Node(name="C", t=t)
    pa_value2 = np.array([3.0, 2.0, 1.0])
    ch_node2 = Node(name="C", t=t+1)
    ch_value2 = np.array([6.0, 5.0, 4.0])

    result = prior_trans.normal_ops(pa_node2, pa_value2, ch_node2, ch_value2, funcs)
    assert isinstance(result, hDict)
    assert isinstance(result[(pa_node2.name, )][t, 0], GPRegression)
    assert isinstance(result[(pa_node.name, )][t, 0], GPRegression)
    assert result[(pa_node.name, )][t+1, 0] is None

def test_collider_ops(setup_prior_trans):
    prior_trans = setup_prior_trans
    pa_nodes = [Node(name="parent1", t=0), Node(name="parent2", t=0)]
    pa_values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    ch_node = Node(name="child", t=1)
    ch_value = np.array([7.0, 8.0, 9.0])
    funcs = hDict(variables=[], nT=3, nTrials=1)
    
    result = prior_trans.collider_ops(pa_nodes, pa_values, ch_node, ch_value, funcs)
    assert isinstance(result, hDict)
    assert isinstance(result[tuple(pa_node.name for pa_node in pa_nodes)][0, 0], GPRegression)
    
    pa_nodes2 = np.array([Node(name="parent3", t=1), Node(name="parent4", t=1), Node(name="parent5", t=1)])
    pa_values2 = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]])
    ch_node2 = Node(name="child", t=2)
    ch_value2 = np.array([7.0, 8.0, 9.0])

    result = prior_trans.collider_ops(pa_nodes2, pa_values2, ch_node2, ch_value2, funcs)
    assert isinstance(result, hDict)
    assert isinstance(result[tuple(pa_node.name for pa_node in pa_nodes)][0, 0], GPRegression)
    assert isinstance(result[tuple(pa_node.name for pa_node in pa_nodes2)][1, 0], GPRegression)


def test_get_M(setup_prior_trans):
    prior_trans = setup_prior_trans
    _, result = prior_trans.get_M()
    
    # [A_0 B_0 C_0 A_1 B_1 C_1 A_2 B_2 C_2]
    expected = np.array(
        [[0., 0., 0., 0., 1., 1., 0., 0., 0.,],
         [0., 0., 0., 0., 0., 1., 0., 0., 0.,],
         [0., 0., 0., 0., 0., 0., 0., 0., 1.,],
         [0., 0., 0., 0., 0., 0., 1., 0., 1.,],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.,],
         [0., 0., 0., 0., 0., 0., 0., 1., 1.,],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.,],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.,],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.,],]
    )
    
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, expected)

def test_fit(setup_prior_trans):
    prior_trans = setup_prior_trans
    data = hDict(variables=prior_trans.variables, nT=3, nTrials=1, default=lambda x, y: np.random.rand(x, y))
    haha = prior_trans.fit(data)
    
    arr = [
        ((Var('A'), 0, Var('B')), [[GPRegression], [None], [None]]), # A_0 -> B_1
        ((Var('A'), 1, Var('C')), [[GPRegression], [None], [None]]), # A_0 -> C_1
        ((Var('A'), 0, Var('A')), [[None], [GPRegression], [None]]), # A_1 -> A_2
        ((Var('A'), 1, Var('B')), [[None], [GPRegression], [None]]), # A_1 -> B_2
        ((Var('C'), 1, Var('B')), [[None], [GPRegression], [None]]), # C_1 -> B_2
        ((Var('C'), 0, Var('C')), [[None], [GPRegression], [None]]), # C_1 -> C_2        
        ((Var('B'),), [[GPRegression], [None], [None]]), # B_0 -> B_2
        ((Var('C'),), [[GPRegression], [None], [None]]), # C_0 -> C_2
        ((Var('A'), Var('C')), [[GPRegression], [None], [None]]), # A_0 -> C_1, B_0 -> C_1
        ((Var('A'), Var('B'), Var('C')), [[None], [GPRegression], [None]]), # A_1 -> B_2, C_1 -> B_2, B_0 -> B_2
    ]
    '''
    transition edges
        ("A_0", "B_1"), c1
        ("A_0", "C_1"), c1, collider 2
        ("B_0", "B_2"), x, collider1
        ("A_1", "B_2"), c3, collider1
        ("C_0", "C_1"), x, collider 2
        ("C_1", "C_2"), c2
        ("A_1", "A_2"), c3
        ("C_1", "B_2"), c2, collider1

    '''
    print(haha)   
    for item in arr:
        key, value = item
        for t in range(3):
            if value[t][0] is not None:
                assert isinstance(haha[key][t][0], value[t][0]) 
            else:
                assert haha[key][t][0] is None