from sems.syn import X2Y, Y2X, Confounder
from data_struct import hDict, Var
from models import MoG, GLMTanh
from utils.metrics import PDC, InfoGain
from sems import SEMBase
from src.run_bf import HypothesisTesting
from data_ops import DSamplerObsBF, DatasetBF
import numpy as np
import pytest

def test_hypothesis_testing_init():
    sem = Confounder()
    variables = [Var("X"), Var("U"), Var("Y")]
    varX = Var("X")
    varY = Var("Y")
    n_pdc = 200
    dtype = "float32"
    epsilon = hDict(
        variables=variables,
        nT=1,
        nTrials=3000,
        default=lambda x, y: np.random.randn(x, y).astype(dtype),
    )
    criterion = PDC(0.2, 10, 0.1, dtype=dtype)
    m_obs_0 = MoG(n_dist=2, dtype=dtype)
    m_obs_1 = MoG(n_dist=2, dtype=dtype, link_fn=GLMTanh(dtype=dtype))

    ht = HypothesisTesting(sem, variables, varX, varY, n_pdc, epsilon, criterion, m_obs_0, m_obs_1, dtype)
    
    assert ht.D_Obs.n_samples == epsilon.nTrials


@pytest.mark.parametrize(
    "varX,varY,sem,variables", [(Var("X"), Var("X"), Confounder(), [Var("X"), Var("U"), Var("Y")]), 
                      (Var("X"), Var("Z"), Confounder(), [Var("X"), Var("U"), Var("Y")]),
                      (Var("Z"), Var("Y"), Confounder(), [Var("X"), Var("U"), Var("Y")]),
                        (Var("X"), Var("Y"), X2Y(), [Var("X"), Var("U"), Var("Y")]),
                      ]
)
def test_hypothesis_testing_init_invalid(varX, varY, sem, variables):
    n_pdc = 200
    dtype = "float32"
    epsilon = hDict(
        variables=variables,
        nT=1,
        nTrials=3000,
        default=lambda x, y: np.random.randn(x, y).astype(dtype),
    )
    criterion = PDC(0.2, 10, 0.1, dtype=dtype)
    m_obs_0 = MoG(n_dist=2, dtype=dtype)
    m_obs_1 = MoG(n_dist=2, dtype=dtype, link_fn=GLMTanh(dtype=dtype))

    
    with pytest.raises(AssertionError):
        ht = HypothesisTesting(sem, variables, varX, varY, n_pdc, epsilon, criterion, m_obs_0, m_obs_1, dtype)

def test_hypothesis_testing_obs():
    sem = Confounder()
    variables = [Var("X"), Var("U"), Var("Y")]
    varX = Var("X")
    varY = Var("Y")
    n_pdc = 200
    dtype = "float32"
    epsilon = hDict(
        variables=variables,
        nT=1,
        nTrials=3000,
        default=lambda x, y: np.random.randn(x, y).astype(dtype),
    )
    criterion = PDC(0.2, 10, 0.1, dtype=dtype)
    m_obs_0 = MoG(n_dist=2, dtype=dtype)
    m_obs_1 = MoG(n_dist=2, dtype=dtype, link_fn=GLMTanh(dtype=dtype))

    ht = HypothesisTesting(sem, variables, varX, varY, n_pdc, epsilon, criterion, m_obs_0, m_obs_1, dtype)
    genObsY, D_Obs = ht.obs(sem, variables, varX, varY, epsilon, dtype)
    
    assert isinstance(genObsY, DSamplerObsBF)
    assert isinstance(D_Obs, DatasetBF)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_hypothesis_testing_run(dtype):
    sem = Confounder()
    variables = [Var("X"), Var("U"), Var("Y")]
    varX = Var("X")
    varY = Var("Y")
    n_pdc = 200
    epsilon = hDict(
        variables=variables,
        nT=1,
        nTrials=3000,
        default=lambda x, y: np.random.randn(x, y).astype(dtype),
    )
    criterion = PDC(0.2, 10, 0.1, dtype=dtype)
    m_obs_0 = MoG(n_dist=2, dtype=dtype)
    m_obs_1 = MoG(n_dist=2, dtype=dtype, link_fn=GLMTanh(dtype=dtype))

    ht = HypothesisTesting(sem, variables, varX, varY, n_pdc, epsilon, criterion, m_obs_0, m_obs_1, dtype)
    logs = ht.run(5)
    
    assert len(logs) == 5
    for log in logs:
        assert isinstance(log, (float, np.float32, np.float64))


