import pytest
import numpy as np
from data_struct import IntervLog, Var, Node

def test_intervlog_initialization():
    exp_sets = [(Var('X'),), (Var('Z'),), (Var('X'), Var('Z'))]
    il = IntervLog(exp_sets=exp_sets, nT=4, nTrials=5)
    assert il.nT == 4
    assert il.nTrials == 5
    assert il.set_idx == {exp_sets[0]: 0, exp_sets[1]: 1, exp_sets[2]: 2}
    assert il.data.shape == (4, 5, 3, 4)

@pytest.mark.parametrize("exp_sets", [
    [],
    [(Var('X'),), (Var('X'), Var('Y')), (Node('X', 0), Var('Y'))],
    [(Var('X'),), (Var('X'), Var('Y')), ()]
])
def test_intervlog_invalid(exp_sets):

    with pytest.raises(AssertionError):
        il = IntervLog(exp_sets=exp_sets, nT=4, nTrials=5)

@pytest.mark.parametrize("t", [0,3])
@pytest.mark.parametrize("trial", [1,3])
def test_get_value(t,trial):
    exp_sets = [(Var('X'),), (Var('Z'),), (Var('X'), Var('Z'))]
    il = IntervLog(exp_sets=exp_sets, nT=4, nTrials=5)

    for i in range(3):
        il.update(0, 2, impv=10*-i, y_values=100*i, i_set=exp_sets[i], i_level=np.array([1]*len(exp_sets[i])))
    il.update(t, 0, impv=20, y_values=200, i_set=exp_sets[1], i_level=np.array([2]))
    il.update(0, trial, impv=30, y_values=300, i_set=exp_sets[1], i_level=np.array([2]))
    il.update(t, trial, impv=40, y_values=400, i_set=exp_sets[1], i_level=np.array([2]))
    il.update(t, trial, impv=40, y_values=50, i_set=exp_sets[1], i_level=np.array([2]))
    il.update(t, trial, impv=10, y_values=40, i_set=exp_sets[2], i_level=np.array([2,2]))

    assert il.get_value(il.data[0, 2], key=0, direction=max)[0] == 0
    assert il.get_value(il.data[0, 2], key=0, direction=min)[0] == -20
    assert il.get_value(il.data[0, 2], key=3, direction=max)[0] == -20
    assert il.get_value(il.data[0, 2], key=3, direction=min)[0] == 0
    
    assert il.get_value(il.data[0, trial], key=0, direction=max)[1] == exp_sets[1]
    assert all(il.get_value(il.data[t, 0], key=0, direction=max)[2] == np.array([2]))
    assert il.get_value(il.data[2, 2], key=3, direction=min) == None
    assert il.get_value(il.data[t, trial], key=3, direction=max)[3] == 50


@pytest.mark.parametrize("t,trial", [(0,2),(1,3),(1,4)])
def test_opt(t, trial):
    exp_sets = [(Var('X'),), (Var('Z'),), (Var('X'), Var('Z'))]
    il = IntervLog(exp_sets=exp_sets, nT=4, nTrials=5)
    il.update(t, trial, impv=30, y_values=300, i_set=exp_sets[0], i_level=np.array([1]))
    il.update(t, trial, impv=20, y_values=200, i_set=exp_sets[1], i_level=np.array([2]))
    il.update(t, trial, impv=10, y_values=100, i_set=exp_sets[2], i_level=np.array([2,2]))
    assert il.opt_y_trial[t,trial][3] == 100
    assert all(x == None for x in il.opt_y_trial[1,0])
    assert il.opt_impv_trial[t,trial][3] == 300
    assert all(x == None for x in il.opt_impv_trial[1,0])
    assert il.sol[t][3] == 100
    assert all(x == None for x in il.sol[2])

@pytest.mark.parametrize("t,trial", [(0,2),(1,3),(1,4)])
def test_update_y(t,trial):
    exp_sets = [(Var('X'),), (Var('Z'),), (Var('X'), Var('Z'))]
    il = IntervLog(exp_sets=exp_sets, nT=4, nTrials=5)
    il.update(t, trial, impv=10, y_values=200, i_set=exp_sets[0], i_level=np.array([1]))
    il.update(t, trial, impv=10, y_values=200, i_set=exp_sets[1], i_level=np.array([1]))
    il.update_y(t, trial, exp_sets[0], 100)
    assert il.data[t, trial, 0][3] == 100
    assert il.data[t, trial, 0][0] == 10
    assert il.data[t, trial, 1][3] == 200


def test_intervlog_initialization():
    exp_sets = [(Var('X'),), (Var('Z'),), (Var('X'), Var('Z'))]
    il = IntervLog(exp_sets=exp_sets, nT=4, nTrials=5)
    assert il.nT == 4
    assert il.nTrials == 5
    assert il.set_idx == {exp_sets[0]: 0, exp_sets[1]: 1, exp_sets[2]: 2}
    assert il.data.shape == (4, 5, 3, 4)

    
@pytest.mark.parametrize("t,trial,impv,y_values,i_set,i_level", [
    (0, 0, 10, 100, (Var('X'),), np.array([1])),
    (0, 1, 20, 200, (Var('Z'),), np.array([2])),
    (1, 2, 30, 300, (Var('X'), Var('Z')), np.array([3, 3])),
    (2, 3, 40, 400, (Var('X'),), np.array([4])),
    (3, 4, 50, 500, (Var('Z'),), np.array([5]))
])
def test_update(t, trial, impv, y_values, i_set, i_level):
    exp_sets = [(Var('X'),), (Var('Z'),), (Var('X'), Var('Z'))]
    il = IntervLog(exp_sets=exp_sets, nT=4, nTrials=5)
    il.update(t, trial, impv=impv, y_values=y_values, i_set=i_set, i_level=i_level)
    assert il.data[t, trial, il.set_idx[i_set]][0] == impv
    assert il.data[t, trial, il.set_idx[i_set]][1] == i_set
    assert np.array_equal(il.data[t, trial, il.set_idx[i_set]][2], i_level)
    assert il.data[t, trial, il.set_idx[i_set]][3] == y_values

@pytest.mark.parametrize("t,trial,impv,y_values,i_set,i_level", [
    (-1, 0, 10, 100, (Var('X'),), np.array([1])),
    (0, -1, 20, 200, (Var('Z'),), np.array([2])),
    (4, 2, 30, 300, (Var('X'), Var('Z')), np.array([3, 3])),
    (2, 5, 40, 400, (Var('X'),), np.array([4])),
    (3, 4, '50', 500, (Var('Z'),), np.array([5])),
    (3, 4, 50, '500', (Var('Z'),), np.array([5])),
    (3, 4, 50, 500, 'XZ', np.array([5])),
    (3, 4, 50, 500, (), np.array([5])),
    (3, 4, 50, 500, ("X"), np.array([5])),
    (3, 4, 50, 500, (Var('X'),), [5]),
    (3, 4, 50, 500, (Var('X'),), np.array([5, 5]))
])
def test_update_invalid(t, trial, impv, y_values, i_set, i_level):
    exp_sets = [(Var('X'),), (Var('Z'),), (Var('X'), Var('Z'))]
    il = IntervLog(exp_sets=exp_sets, nT=4, nTrials=5)
    with pytest.raises(AssertionError):
        il.update(t, trial, impv=impv, y_values=y_values, i_set=i_set, i_level=i_level)