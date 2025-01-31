import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import numpy as np
from utils.graphs import get_generic_graph
from sems.toy_sems import StationaryIndependentSEM
from sems import SEMHat
from utils.tools import powerset
from utils.grids import get_interv_sampler
from data_struct import hDict, Var, GraphObj, Node, esDict, IntervLog
from data_ops import DatasetObsDCBO, DSamplerObsDCBO, DSamplerInv, DatasetInv
from surrogates import PriorEmit, PriorTrans, Surrogate
from bo import ManualCausalEI, CausalEI, BOModel
from bo import FixedCost
from copy import deepcopy
# define graph

import tensorflow as tf

tf.config.run_functions_eagerly(True)


nT = 3
variables = [Var("X"), Var("Z"), Var("Y")]
target_variable = Var("Y")
# define SEM
sem = StationaryIndependentSEM()
n_samples = 6

# define exploration sets
exploration_sets = list(powerset([Var("X"), Var("Z")]))

# define observation dataset
epsilon = hDict(variables=variables, nT=nT, nTrials=n_samples, default=np.random.randn)
initial_values = hDict(variables=variables)
interv_levels = hDict(variables=variables, nT=nT)

genObsY = DSamplerObsDCBO(
    sem=sem,
    nT=nT,
    variables=variables,
)

obsY = genObsY.sample(
    initial_values=initial_values,
    interv_levels=interv_levels,
    epsilon=epsilon,
    n_samples=n_samples,
)


D_O = DatasetObsDCBO(
    nT=nT,
    n_samples=n_samples,
    initial_values=initial_values,
    interv_levels=interv_levels,
    epsilon=epsilon,
    dataY=obsY,
)

# define intervention domain
intervention_domain = {Var("X"): [-4, 1], Var("Z"): [-3, 3]}
interv_sampler = get_interv_sampler(exploration_sets, intervention_domain)

# TODO: nodes are not in Var / Node type
# TODO : TOPOLOGICALLY ORDERED DICT for SEM, because SEM has orders
G = get_generic_graph(
    start_time=0,
    stop_time=nT,
    topology="independent",
    nodes=["X", "Z", "Y"],
    target_node="Y",
)


print(G)

"""
digraph { rankdir=LR; X_0 -> Y_0; Z_0 -> Y_0; X_1 -> Y_1; Z_1 -> Y_1; X_2 -> Y_2; Z_2 -> Y_2; X_3 -> Y_3; Z_3 -> Y_3;  X_0 -> X_1; Z_0 -> Z_1; Y_0 -> Y_1; X_1 -> X_2; Z_1 -> Z_2; Y_1 -> Y_2; X_2 -> X_3; Z_2 -> Z_3; Y_2 -> Y_3;  { rank=same; X_0 Z_0 Y_0 } { rank=same; X_1 Z_1 Y_1 } { rank=same; X_2 Z_2 Y_2 } { rank=same; X_3 Z_3 Y_3 }  }
"""

G = GraphObj(graph=G, nT=nT, target_var=target_variable)

prior_emit = PriorEmit(G)
prior_trans = PriorTrans(G)

prior_emit.fit(D_O.dataY)
prior_trans.fit(D_O.dataY)

semhat = SEMHat(G, prior_emit, prior_trans)
print(semhat.get_edgekeys(Node("Y", 1), 1))
print(semhat.get_edgekeys(Node("Y", 1), 0))

init_tvalue = 0
N = 10
outcome_values = hDict(
    variables=G.variables,
    nT=nT,
    nTrials=N,
)

surr = Surrogate(semhat)
D_Inv = DatasetInv(nT=nT, exp_sets=exploration_sets)


# stores interventional data levels
invDX = esDict(exp_sets=exploration_sets, nT=nT, nTrials=N)

# create intervention blanket
genInvY = DSamplerInv(semhat)
invLogger = IntervLog(exp_sets=exploration_sets, nT=nT, nTrials=N)
bo_models = hDict(
    variables=exploration_sets,
    nT=nT,
    nTrials=1,
)
opt_ilvl = hDict(
    variables=G.variables,
    nT=nT,
    nTrials=1,
)
static_epsilon = hDict(
    variables=variables, nT=nT, nTrials=1, default=lambda x, y: np.zeros((x, y))
)

# TODO: remove target_variable from variables
acq_cost = FixedCost(variables, 1)

for t in range(nT):

    for trial in range(N):

        for es in exploration_sets:
            mean_f, variance_f = surr.create(t=t, interv_levels=opt_ilvl, es=es)

            if bo_models[es][t, 0] is None:
                acq = ManualCausalEI(
                    target_variable,
                    mean_f,
                    variance_f,
                    init_tvalue,
                )
            else:
                acq = CausalEI(
                    bo_model=bo_models[es][t, 0],
                )
            
            invDX[es][t, :] = temp = interv_sampler[es].sample(N)

            improvements = acq.evaluate(temp.astype(np.float64)) / acq_cost.evaluate(es, temp)        
            invLogger.update(
                t=t,
                trial=trial,
                i_set=es,
                i_level=invDX[es][t, (idx := np.argmax(improvements))],
                impv=improvements[idx],
                y_values=None,
            )

        # update the optimal intervention set per trial
        trial_ilvl = deepcopy(opt_ilvl)
        _, trial_es, trial_lvl, _ = invLogger.opt_ptrial[t, trial]

        for vid, var in enumerate(trial_es):
            trial_ilvl[var][t, 0] = trial_lvl[vid]
        
        
        # get the y value for the optimal intervention set
        y_new = genObsY.sample(interv_levels=trial_ilvl, epsilon=static_epsilon, n_samples=1)[target_variable][t][0]  # (1x1)
        D_Inv.update(
            t=t,
            es=trial_es,
            x=trial_lvl,
            y=y_new,
        )
        invLogger.update_y(t, trial, trial_es, y_new)

        # update the BO model
        dataIX, dataIY = D_Inv.get(trial_es, t)
        if bo_models[trial_es][t, 0] is None:
            # TODO: model.likelihood.variance.fix() is unsolved
            # should be corresponding to log_prob of the GP model
            mean_f, variance_f = surr.create(t=t, interv_levels=trial_ilvl, es=trial_es)
            
            bo_models[trial_es][t, 0] = BOModel(
                es,
                target_variable,
                mean_f,
                variance_f,
            )

        bo_models[trial_es][t, 0].fit(dataIX, dataIY.reshape(-1))


    # update opt_ilvl
    _, best_es, best_lvl, _ = invLogger.sol[t]
    for vid, var in enumerate(best_es):
        opt_ilvl[var][t, 0] = best_lvl[vid]
    
# DCBO.run ==>  oi_set, oi_level
print(invLogger.sol)

