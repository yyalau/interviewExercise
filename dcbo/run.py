import numpy as np
from utils.graphs import get_generic_graph
from sems.toy_sems import StationaryIndependentSEM
from sems import SEMHat
from utils.tools import powerset
from utils.grids import get_interv_grids
from data_struct import hDict, Var, GraphObj, Node
from data_ops import DatasetObs, DSamplerObs, DSamplerInv
from surrogates import PriorEmit, PriorTrans, Surrogate
# define graph

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[5], 'GPU')


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

obsY = DSamplerObs(
    sem = sem, nT = nT, variables = variables, 
).sample(    
    initial_values = initial_values,
    interv_levels = interv_levels,
    epsilon = epsilon,
    n_samples = n_samples 
)


D_O = DatasetObs(
    nT = nT,
    n_samples = n_samples,
    initial_values=initial_values,
    interv_levels=interv_levels,
    epsilon=epsilon,
    dataY=obsY
)

# define intervention domain
intervention_domain = {Var("X"): [-4, 1], Var("Z"): [-3, 3]}
interv_grids = get_interv_grids(exploration_sets, intervention_domain, intervals=100)

# TODO: nodes are not in Nodes
G = get_generic_graph(
        start_time=0,
        stop_time=nT,
        topology="independent",
        nodes=["X", "Z", "Y"],
        target_node="Y",
)
'''
G = digraph { rankdir=LR; X_0 -> Y_0; Z_0 -> Y_0; X_1 -> Y_1; Z_1 -> Y_1; X_2 -> Y_2; Z_2 -> Y_2; X_3 -> Y_3; Z_3 -> Y_3;  X_0 -> X_1; Z_0 -> Z_1; Y_0 -> Y_1; X_1 -> X_2; Z_1 -> Z_2; Y_1 -> Y_2; X_2 -> X_3; Z_2 -> Z_3; Y_2 -> Y_3;  { rank=same; X_0 Z_0 Y_0 } { rank=same; X_1 Z_1 Y_1 } { rank=same; X_2 Z_2 Y_2 } { rank=same; X_3 Z_3 Y_3 }  }
'''
print(G)


G = GraphObj(
    graph=G,
    nT=nT
)

prior_emit = PriorEmit(G)
prior_trans = PriorTrans(G)

prior_emit.fit(D_O.dataY)
prior_trans.fit(D_O.dataY)

# import ipdb; ipdb.set_trace()
semhat = SEMHat(G, prior_emit, prior_trans)
print(semhat.get_edgekeys(Node("Y", 1), 1))
print(semhat.get_edgekeys(Node("Y", 1), 0))


mean, variance = Surrogate(semhat).create(
    initial_values=initial_values,
    interv_levels=interv_levels,
    n_samples=n_samples
)

print(mean(), variance())

# define acquisiton function

# add new data to intervention datatset (using target_functions)

# define BO optimization

# record optimal intervention hat set / level / y_values
    
# number of trials for each time step
N = 10
# DCBO.run ==>  oi_set, oi_level

