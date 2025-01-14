import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import numpy as np
from utils.graphs import get_generic_graph
from sems.toy_sems import StationaryIndependentSEM
from sems import SEMHat
from utils.tools import powerset
from utils.grids import get_interv_sampler
from data_struct import hDict, Var, GraphObj, Node, esDict, IntervLog
from data_ops import DatasetObs, DSamplerObs, DSamplerInv, DatasetInv
from surrogates import PriorEmit, PriorTrans, Surrogate
from bo import ManualCausalEI, CausalEI, BOModel
from models import CausalRBF
# define graph

import tensorflow as tf



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

genObsY = DSamplerObs(
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


D_O = DatasetObs(
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

# nodes are not in Var / Node type
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

# import ipdb; ipdb.set_trace()
semhat = SEMHat(G, prior_emit, prior_trans)
print(semhat.get_edgekeys(Node("Y", 1), 1))
print(semhat.get_edgekeys(Node("Y", 1), 0))

init_tvalue = 1000000
N = 10
outcome_values = hDict(
    variables=G.variables,
    nT=nT,
    nTrials=N,
)

surr = Surrogate(semhat)
# stores interventional data levels


# ================================ t = 0, trial = 0 ================================
# TODO : ORDERED DICT for SEM, because SEM has orders
t = 0
invDX = esDict(exp_sets=exploration_sets, nT=nT, nTrials=N)
for es in exploration_sets:
    # sample from uniform distribution
    invDX[es][t,:] = interv_sampler[es].sample(N)

# create intervention blanket
genInvY = DSamplerInv(semhat)
invLogger = IntervLog()

opt_ilvl = hDict(variables=G.variables, nT=nT, nTrials=1,)

for es in exploration_sets:

    # inputs N sampled points from interv domain
    # temp_ilvl =  hDict(variables=variables, nT=nT, nTrials=N)
    # for vid, var in enumerate(es):
    #     temp_ilvl[var][t,:] = invDX[es][t,:, vid]    
    
    mean_f, variance_f = surr.create(t = t, interv_levels=opt_ilvl, es = es)    
    acq = ManualCausalEI(target_variable, mean_f, variance_f, )
    improvements = acq.evaluate(invDX[es][t], cmin = init_tvalue)
    
    invLogger.update(
        t = 0,
        i_set=es,
        i_level= invDX[es][t,(idx := np.argmax(improvements))],
        impv = improvements[idx],
        y_values=None,
    )

# update the optimal intervention
_, best_es, best_lvl, _ = invLogger.get_opt()[t]
for vid, var in enumerate(best_es):
    opt_ilvl[var][t,0] = best_lvl[vid]

# self.interventional_data_x[temporal_index][exploration_set]
static_epsilon = hDict(variables=variables, nT=nT, nTrials=1, default=lambda x, y: np.random.randn(x, y))
y_new = genObsY.sample(interv_levels= opt_ilvl,
                       epsilon=static_epsilon,
                       n_samples=1)[target_variable][t][0] #(1x1)

D_Inv = DatasetInv(nT = nT, exp_sets=exploration_sets)
D_Inv.update(
    t = t,
    es = best_es,
    x = best_lvl,
    y = y_new,
)

# ================================ t = 0, trial = 1 ================================

bo_models = hDict(variables=exploration_sets, nT=nT, nTrials=1,)

# assert self.interventional_data_x[temporal_index][exploration_set] is not None
# assert self.interventional_data_y[temporal_index][exploration_set] is not None

for es in exploration_sets[0:]:
    if bo_models[es][t,0] is None:
        

        mean_f, variance_f = surr.create(t = t, interv_levels=opt_ilvl, es=es)        
        # TODO: model.likelihood.variance.fix() is unsolved
        # should be corresponding to log_prob of the GP model
        bo_models[es][t,0] = BOModel(es, target_variable, mean_f, variance_f, )


# record optimal intervention hat set / level / y_values

# add new data to intervention datatset (using target_functions)


# number of trials for each time step
# DCBO.run ==>  oi_set, oi_level
