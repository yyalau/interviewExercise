import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from sems.syn import X2Y, Y2X, Confounder
from data_ops import DatasetBF, DSamplerObsBF, DSamplerInv, DatasetInv
from data_struct import hDict, Var
import numpy as np
from models import MoG, GLMTanh
from utils.tools import get_probH0, update_posteriorH0
from utils.grids import get_interv_sampler
from utils.metrics import PDC, InfoGain
from collections import OrderedDict

variables = [Var("X"), Var("U"), Var("Y")]
varX = Var("X")
varY = Var("Y")
sem = Confounder()


n_obs = 3000
n_pdc = 200
n_dist = 2
beta = 0.2
k0 = 10
k1 = 1./k0
dtype = "float32"
n_trials = 10


epsilon = hDict(variables=variables, nT=1, nTrials=n_obs, default=lambda x,y : np.random.randn(x, y).astype(dtype))

genObsY = DSamplerObsBF(
    sem=sem,
    variables=variables,
    dtype=dtype
)

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

D_Int = DatasetBF(dtype=dtype)


# Estimate the interventional distribution for the hypothese m^hat_0, m^hat_1 with the observational data D_obs as described in 3.2
m_obs_0 = MoG(n_dist=n_dist, dtype = dtype)
m_obs_0.fit(D_Obs.dataX, D_Obs.dataY, verbose=False)


m_obs_1 = MoG(n_dist=n_dist, dtype = dtype, link_fn=GLMTanh(dtype = dtype))
m_obs_1.fit(D_Obs.dataX, D_Obs.dataY, verbose=False)


# pdf_H0 = m_obs_0.prob(D_Obs.dataY)
# pdf_H1 = m_obs_1.prob(D_Obs.dataY, x = D_Obs.dataX)
# prior_H0 = tf.reduce_mean(tf.cast(pdf_H0 > pdf_H1, dtype=dtype)).numpy()
prior_H0 = get_probH0(D_Obs.dataX, D_Obs.dataY, m_obs_0, m_obs_1, dtype = dtype)

posterior_H0 = 0.5

print("P(H_0) = ", prior_H0, "; P(H_1) = ", 1-prior_H0)

for trial in range(n_trials):

    # Estimate PDC (Dint, do(X = x)) using Monte Carlo method as described in Eq. (8)
    # Optimize x_opt to maximize PDC (Dint, do(X = x_opt)) [eq. (7)] (x should be a parameter of the optimization)
    
    # pdc = PDC(beta, k0, k1, dtype=dtype)
    # x_opt = pdc.fit(prior_H0, D_Int, m_obs_0, m_obs_1, n_samples = n_pdc, n_restart=100, verbose=True).get_xopt()
    ig = InfoGain(dtype=dtype)
    x_opt = ig.fit(prior_H0, D_Int, m_obs_0, m_obs_1, n_samples = n_pdc, n_restart = 100, verbose = False).get_xopt()

    # drawn from the probability distribution p_G as in D_obs
    
    temp = hDict(variables=variables, nT=1, nTrials=1, default=lambda x,y : np.zeros((x, y)).astype(dtype))
    temp[varX] = x_opt; temp[varY] = np.zeros((1,1))
    
    y_new = genObsY.sample(epsilon = temp,
                        n_samples = 1)

    D_Int.update_new(y_new[varX], y_new[varY])

    # P(D_int | H_0) = likelihood = m0.prob(D_int, x= x_opt)
    posterior_H0 = m_obs_0.prob(D_Int.dataY)
    posterior_H1 = m_obs_1.prob(D_Int.dataY, x = D_Int.dataX)

    posterior_H0 = update_posteriorH0(D_Int, m_obs_0, m_obs_1, prior_H0)

    print("trial: ", trial, "; P(H_0 | D_int) = ", posterior_H0, "; P(H_1 | D_int) = ", 1-posterior_H0)