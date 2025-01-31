import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from sems.syn import X2Y, Y2X, Confounder
from data_ops import DatasetBF, DSamplerObsBF, DSamplerInv, DatasetInv
from data_struct import hDict, Var
import numpy as np
from models import MoG, GLMTanh
from utils.grids import get_interv_sampler
from utils.metrics import PDC

variables = [Var("X"), Var("Y")]
varX = Var("X")
varY = Var("Y")
sem = X2Y()

genObsY = DSamplerObsBF(
    sem=sem,
    variables=variables,
)

n_obs = 3000
n_pdc = 200
n_dist = 2
beta = 0.2
k0 = 10
k1 = 1./k0
dtype = "float32"


epsilon = hDict(variables=variables, nT=1, nTrials=n_obs, default=lambda x,y : np.random.randn(x, y).astype(dtype))
obsY = genObsY.sample(
    epsilon=epsilon,
    n_samples=n_obs
)
D_Obs = DatasetBF(
    n_samples=n_obs,
    epsilon=epsilon,
    dataX = epsilon[varX][0].reshape(-1, 1),
    dataY=obsY[varY][0].reshape(-1, 1),
    dtype = dtype
)

M = 10 # intervention sample size

exp_set = [(varX,)]
# intervention_domain = {Var("X"): [-5, 5], Var("Z"): [-3, 3]}
# x_sampler = get_interv_sampler(exp_set, intervention_domain)[exp_set[0]]

D_Int = DatasetBF()



# Estimate the interventional distribution for the hypothese m^hat_0, m^hat_1 with the observational data D_obs as described in 3.2
m_obs_0 = MoG(n_dist=n_dist, dtype = D_Obs.dtype)
m_obs_0.fit(D_Obs.dataX, D_Obs.dataY, verbose=False)

m_obs_1 = MoG(n_dist=n_dist, dtype = D_Obs.dtype, link_fn=GLMTanh(dtype = obsY[varY].dtype))
m_obs_1.fit(D_Obs.dataX, D_Obs.dataY, verbose=False)


pdf_H0 = m_obs_0.prob(D_Obs.dataY)
pdf_H1 = m_obs_1.prob(D_Obs.dataY, x = D_Obs.dataX)

prior_H0 = tf.reduce_mean(tf.cast(pdf_H0 > pdf_H1, "float")).numpy()
prior_H1 = 1 - prior_H0

print("P(H_0) = ", prior_H0, "; P(H_1) = ", prior_H1)

# Estimate PDC (Dint, do(X = x)) using Monte Carlo method as described in Eq. (8)
# Optimize x_opt to maximize PDC (Dint, do(X = x_opt)) [eq. (7)] (x should be a parameter of the optimization)
pdc = PDC(beta, k0, k1, dtype=D_Obs.dtype)
x_opt = pdc.fit(prior_H0, D_Int, m_obs_0, m_obs_1, n_samples = n_pdc, n_restart=100, verbose=True).get_xopt()

# drawn from the probability distribution p_G as in D_obs
y_new = genObsY.sample(epsilon = x_opt, 
                       n_samples = 1)

# P(D_int | H_0) = likelihood = m0.prob(D_int, x= x_opt)

