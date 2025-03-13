import pytest
import numpy as np
import tensorflow as tf
import copy
from data_struct.newdict import newDict, hDict, esDict
from data_struct.node import Var

def test_newdict_init():
    arr = [(Var('a'), np.array([1, 2, 3])), (Var('b'), np.array([4, 5, 6]))]
    nd = newDict(arr, 3, 2)
    assert nd.arr == arr
    assert list(nd.keys()) == [Var('a'), Var('b')]

    for i, v in enumerate(nd.values()):
        assert np.array_equal(v, arr[i][1])
        


def test_newdict_deepcopy():
    arr = [(Var('a'), np.array([1, 2, 3])), (Var('b'), np.array([4, 5, 6]))]
    nd = newDict(arr, 3, 2)
    nd_copy = copy.deepcopy(nd)

    assert nd is not nd_copy
    for k, v in nd.items():
        assert np.array_equal(v,nd_copy[k])
    
    for k, v in vars(nd).items():
        if k != 'arr':
            assert getattr(nd, k) == getattr(nd_copy, k)
            continue
        for a, b in zip(nd.arr, nd_copy.arr):
            for i, j in zip(a, b):
                assert np.array_equal(i, j)
            # assert all(a == b)

def test_newdict_add():
    arr = [(Var('a'), np.array([1, 2, 3]))]
    nd = newDict(arr, 3, 2)
    nd.default = lambda nT, nTrials: np.zeros((nT, nTrials))
    nd.add(Var('b'))
    assert Var('a') in nd
    assert np.array_equal(nd[Var('a')], np.array([1, 2, 3]))
    assert Var('b') in nd
    assert np.array_equal(nd[Var('b')], np.zeros((3, 2)))


@pytest.mark.parametrize("arr", [
  ["invalid"],
  [(Var('a'),)],
   [(123, np.array([1, 2, 3]))],
   
])
def test_newdict_invalid_arr_type(arr):
    with pytest.raises(AssertionError):
        newDict(arr, 3, 2)


def test_newdict_invalid_nT():
    arr = [(Var('a'), np.array([1, 2, 3]))]
    with pytest.raises(AssertionError):
        nd = newDict(arr, -1, 2)
        
def test_newdict_invalid_nTrials():
    arr = [(Var('a'), np.array([1, 2, 3]))]
    with pytest.raises(AssertionError):
        nd = newDict(arr, 1, 0)
        
@pytest.mark.parametrize("n_dup", [1,2,3])
def test_newdict_duplicate(n_dup):
    arr = [(Var('a'), np.array([[1, 2], [3, 4]]))]
    nd = newDict(arr, 2, 1)
    nd.duplicate(n_dup)
    assert nd[Var('a')].shape == (2, 2*n_dup)

def test_hdict_init():
    variables = [Var('x'), Var('y')]
    hd = hDict(variables=variables, nT=2, nTrials=3)
    assert set(hd.keys()) == set(variables)
    assert hd.nT == 2
    assert hd.nTrials == 3
    assert hd[variables[0]].shape == (2, 3)


@pytest.mark.parametrize("variables", [
    Var('x'),
    ['hihi'],
    [Var('x'), 'hihi']
])
def test_hdict_invalid_init(variables):
    with pytest.raises(AssertionError):
        hd = hDict(variables=variables, nT=2, nTrials=3)


def test_esdict_init():
    exp_sets = [(Var('x'), Var('y')), (Var('z'),), (Var('x'),) ]
    ed = esDict(exp_sets=exp_sets, nT=2, nTrials=3)
    assert ed.exp_sets == exp_sets
    assert ed.nT == 2
    assert ed.nTrials == 3
    
    assert ed[exp_sets[0]].shape == (2, 3, 2)
    assert ed[exp_sets[1]].shape == (2, 3, 1)
    assert ed[exp_sets[2]].shape == (2, 3, 1) 
    
@pytest.mark.parametrize("variables", [
    [Var('x')],
    [],
    [(Var('x'),), ()],
])
def test_esdict_invalid_init(variables):
    with pytest.raises(AssertionError):
        esDict(exp_sets=variables, nT=2, nTrials=3)

