import pytest
import numpy as np
import tensorflow as tf
from data_struct import hDict, Node, Var, GraphObj
from networkx import DiGraph
from surrogates.semhat import SEMHat
from surrogates.surr import Surrogate
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


@pytest.fixture
def semhat():
    G = DiGraph()
    G.add_edges_from(
        [
            ("A_0", "B_1"),
            ("A_0", "C_1"),
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
    
    dataY = hDict(variables=G.variables, nT=3, nTrials=1, default = lambda x,y: np.random.rand(x,y))
    return SEMHat(G, dataY)


@pytest.fixture
def surrogate(semhat):
    return Surrogate(semhat)


def test_surrogate_init(semhat):
    surrogate = Surrogate(semhat)
    assert surrogate.sem == semhat
    assert surrogate.nT == semhat.nT
    assert all(surrogate.variables == semhat.vs)
    assert surrogate.dtype == "float32"


@pytest.mark.parametrize(
    "interv_levels",
    [
        hDict(variables=((Var("X"),)), nT=3, nTrials=1),
        hDict(variables=(Var("X"), Var("A")), nT=3, nTrials=1),
        hDict(variables=(Var('A'), Var('B'), Var('C')), nT=1, nTrials=1),
    ],
)
def test_create_invalid_interv_levels(surrogate, interv_levels):
    with pytest.raises(AssertionError):
        surrogate.create(0, interv_levels, (Var("A"),), Var(name="target"))

@pytest.mark.parametrize(
    "es",
    [   
        None,
        (),
        (Var("X"),),
        (Var("X"), Var("A")),
    ],
)
def test_create_invalid_es(surrogate, es):
    with pytest.raises(AssertionError):
        surrogate.create(
            0,
            hDict(variables=surrogate.variables, nT=3, nTrials=1),
            es,
            Var(name="A"),
        )


def test_create_invalid_target_var(surrogate):
    with pytest.raises(AssertionError):
        surrogate.create(
            0,
            hDict(variables=surrogate.variables, nT=3, nTrials=1),
            (Var('A'), Var('B')),
            Var(name="invalid"),
        )

def test_create_valid(surrogate):
    interv_levels = hDict(variables=surrogate.variables, nT=surrogate.nT, nTrials=1)
    es = (Var(name="A"),)
    target_var = Var(name="C")
    mean, variance = surrogate.create(0, interv_levels, es, target_var)
    assert callable(mean)
    assert callable(variance)




def mock_source(t, es, n_samples):
    return tf.constant([1.0]*n_samples)

def mock_emit(t, _, emit_keys, sample, n_samples):
    return tf.constant([1.0]*n_samples)

def mock_trans(t, _, emit_keys, sample, n_samples):
    return tf.constant([1.0]*n_samples)


# edit this test
@pytest.mark.parametrize(
    "interv",
    [
        np.random.rand(5),
        np.array([None]*5),
        np.array([np.nan]*5),
    ],
)

@pytest.mark.parametrize(
    "var,t,function",
    [
        (Var(name="A"),0, mock_source),
        (Var(name="C"),2, mock_emit),
        (Var(name="B"),1, mock_trans),
    ]
)
def test_select_value(surrogate, interv,var, t, function):
    n_samples = 5
    samples = hDict(variables=(Var('A'), Var('B'), Var('C')), nT=3, nTrials=n_samples)
    
    result = surrogate.select_value(function, interv, var, t, samples, n_samples)
    if interv[0] is None or np.isnan(interv[0]):
        assert all(result == tf.constant([1.0]*n_samples))
    assert result.shape == (n_samples,)

