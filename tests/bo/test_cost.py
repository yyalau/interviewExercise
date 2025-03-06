import bo.cost as c
import pytest
from data_struct import Var

@pytest.mark.parametrize(
    "costClass",
    [
        c.FixedCost,
        c.RandomCost,
        c.InvValueCost,
    ]
)
@pytest.mark.parametrize(
    "variables",
    [
        [1, 2, 3],
        ["A", "B", "C"],
        [Var("X"), Var("Y"), Var("Z"), Var("W")],
    ],
)

def test_CostBase(costClass, variables):
    assert set(costClass(variables, 2).cost_f.keys()) == set(variables)
