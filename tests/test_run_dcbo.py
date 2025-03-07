import pytest
import numpy as np
from data_struct import hDict, Var, GraphObj
from sems.toy_sems import StationaryDependentSEM, StationaryIndependentSEM
from utils.graphs import get_generic_graph
from run_dcbo import DCBO
from data_struct import IntervLog

@pytest.fixture
def setup_dcbo():
    nTrials = 3
    n_samples = 2
    n_obs = 100
    dtype = "float32"
    nT = 3  

    variables = [Var("X"), Var("Z"), Var("Y")]
    target_variable = Var("Y")
    exp_sets = [(Var("X"),), (Var("Z"),), (Var("X"), Var("Z"))]

    initial_values = hDict(variables=variables)
    epsilon = hDict(variables=variables, nT=nT, nTrials=n_obs)
    for var in variables:
        epsilon[var] = np.random.normal(0, 1, (nT, n_obs))

    dag = get_generic_graph(
        start_time=0,
        stop_time=nT,
        topology="dependent",
        nodes=variables,
        target_node=target_variable,
    )
    G = GraphObj(graph=dag, nT=nT, target_var=target_variable)

    sem = StationaryDependentSEM()
    interv_domain = {Var("X"): [-5, 5], Var("Z"): [-5, 20]}

    dcbo = DCBO(
        nTrials,
        n_samples,
        G,
        sem,
        exp_sets,
        interv_domain,
        initial_values,
        epsilon,
        dtype,
    )

    return dcbo

def test_initialization(setup_dcbo):
    dcbo = setup_dcbo
    assert dcbo.nTrials == 10
    assert dcbo.n_samples == 25
    assert isinstance(dcbo.G, GraphObj)
    assert isinstance(dcbo.sem, StationaryDependentSEM)
    assert dcbo.dtype == "float32"

@pytest.mark.parametrize("dag", [get_generic_graph(
        start_time=1,
        stop_time=3,
        topology="dependent",
        nodes=[Var("X"), Var("Z"), Var("Y")],
        target_node=Var("Y"),
    ), get_generic_graph(
        start_time=0,
        stop_time=3,
        topology="dependent",
        nodes=[Var("X"), Var("Z"), Var("W"), Var("Y")],
        target_node=Var("Y"),
    ),
    ]
                         )
def test_dag(dag,):
    nTrials = 10
    n_samples = 25
    n_obs = 100
    dtype = "float32"
    nT = 3

    variables = [Var("X"), Var("Z"), Var("Y")]
    target_variable = Var("Y")
    exp_sets = [(Var("X"),), (Var("Z"),), (Var("X"), Var("Z"))]

    initial_values = hDict(variables=variables)
    epsilon = hDict(variables=variables, nT=nT, nTrials=n_obs)
    for var in variables:
        epsilon[var] = np.random.normal(0, 1, (nT, n_obs))

    G = GraphObj(graph=dag, nT=nT, target_var=target_variable)

    sem = StationaryDependentSEM()
    interv_domain = {Var("X"): [-5, 5], Var("Z"): [-5, 20]}
    
    
    with pytest.raises(AssertionError):
        dcbo = DCBO(
            nTrials,
            n_samples,
            G,
            sem,
            exp_sets,
            interv_domain,
            initial_values,
            epsilon,
            dtype,
        )

def test_get_gt(setup_dcbo):
    dcbo = setup_dcbo
    gt = dcbo.get_gt()
    assert isinstance(gt, list)
    assert len(gt) == dcbo.nT

def test_get_surr(setup_dcbo):
    dcbo = setup_dcbo
    surr = dcbo.get_surr(dcbo.G, dcbo.datasetO)
    assert surr is not None

def test_run(setup_dcbo):
    dcbo = setup_dcbo
    logger = dcbo.run(None, init_tvalue=10)
    assert logger is not None
    assert isinstance(logger, IntervLog)