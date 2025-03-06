import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from sems.syn import X2Y, Y2X, Confounder
from data_ops import DatasetBF, DSamplerObsBF, DatasetInv
from data_struct import hDict, Var
import numpy as np
from models import MoG, GLMTanh
from utils.tools import get_probH0, update_posteriorH0
from utils.grids import get_interv_sampler
from utils.metrics import PDC, InfoGain
from utils.plots import save_result_bf
from collections import OrderedDict


class HypothesisTesting:
    def __init__(
        self, sem, variables, varX, varY, n_obs, n_pdc, dO_eps, criterion, m_obs_0, m_obs_1, dtype
    ):
        if dO_eps is not None:
            assert dO_eps.nTrials == n_obs, "dO_eps.nTrials must be equal to n_obs"
            
        self.sem = sem
        self.variables = variables
        self.varX = varX
        self.varY = varY
        self.n_obs = n_obs
        self.n_pdc = n_pdc
        self.m_obs_0 = m_obs_0      
        self.m_obs_1 = m_obs_1
        self.genObsY, self.D_Obs = self.obs(sem, variables, n_obs, dO_eps)
        self.D_Int = DatasetBF(dtype=dtype)
        self.criterion = criterion

    def obs(self, sem, variables, n_obs, epsilon):
        genObsY = DSamplerObsBF(sem=sem, variables=variables, dtype=dtype)

        obsY = genObsY.sample(epsilon=epsilon, n_samples=n_obs)
        D_Obs = DatasetBF(
            dataX=epsilon[varX][0].reshape(-1, 1),
            dataY=obsY[varY][0].reshape(-1, 1),
            dtype=dtype,
        )
        return genObsY, D_Obs

    def run(self, n_trials):
        logs = []
        
        self.m_obs_0.fit(self.D_Obs.dataX, self.D_Obs.dataY, verbose=False)
        self.m_obs_1.fit(self.D_Obs.dataX, self.D_Obs.dataY, verbose=False)

        prior_H0 = get_probH0(
            self.D_Obs.dataX, self.D_Obs.dataY, self.m_obs_0, self.m_obs_1, dtype=dtype
        )
        posterior_H0 = 0.5
        print("P(H_0) = ", prior_H0, "; P(H_1) = ", 1 - prior_H0)

        for trial in range(n_trials):

            result = self.criterion.fit(
                prior_H0,
                self.D_Int,
                self.m_obs_0,
                self.m_obs_1,
                n_samples=n_pdc,
                n_restart=10,
                verbose=False,
            )
            x_opt = self.criterion.get_xopt()
            

            # drawn from the probability distribution p_G as in D_obs

            temp = hDict(
                variables=self.variables,
                nT=1,
                nTrials=1,
                default=lambda x, y: np.zeros((x, y)).astype(dtype),
            )
            temp[varX] = x_opt
            temp[varY] = np.zeros((1, 1))

            y_new = self.genObsY.sample(epsilon=temp, n_samples=1)

            self.D_Int.update(y_new[varX], y_new[varY])
            posterior_H0 = update_posteriorH0(
                self.D_Int, self.m_obs_0, self.m_obs_1, prior_H0
            )

            print(
                "trial: ",
                trial,
                "; P(H_0 | D_int) = ",
                posterior_H0,
                "; P(H_1 | D_int) = ",
                1 - posterior_H0,
                "; result = ", result.numpy(),
            )
            logs.append(result.numpy()*-1)

        return logs

if __name__ == "__main__":
    varX = Var("X")
    varY = Var("Y")
    dtype = "float32"    
    n_obs = 3000

    testing = [
        {
            "name": "confounder",
            "sem": Confounder(),
            "variables": [Var("X"), Var("U"), Var("Y")],
            "title": "Confounder: X <- U -> Y",
        },
        {
            "name": "x2y",
            "sem": X2Y(),
            "variables": [Var("X"), Var("Y")],
            "title": "X -> Y",
            
        },
        {
            "name": "y2x",
            "sem": Y2X(),
            "variables": [Var("X"), Var("Y")],
            "title": "Y -> X",
        },
    ]
    n_pdc = 200            
    n_dist = 2
    
    
    beta = 0.2
    k0 = 10
    k1 = 1.0 / k0

    metrics = {
        "pdc": PDC(beta, k0, k1, dtype=dtype),
        "ig": InfoGain(dtype=dtype),
    }

    
    for testcase in testing:
        logs = {}
        for m_name, m_f in metrics.items():
        
            epsilon = hDict(
                variables=testcase['variables'],
                nT=1,
                nTrials=n_obs,
                default=lambda x, y: np.random.randn(x, y).astype(dtype),
            )


            m_obs_0 = MoG(n_dist=n_dist, dtype=dtype)
            m_obs_1 = MoG(n_dist=n_dist, dtype=dtype, link_fn=GLMTanh(dtype=dtype))

            n_trials = 10 # intervention sample size


            ht = HypothesisTesting(
                testcase['sem'], testcase['variables'], varX, varY, n_obs, n_pdc, epsilon, m_f, m_obs_0, m_obs_1, dtype
            )
            logs[m_name] = ht.run(n_trials)
            
        
        name = testcase['name']
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)),  "results", f"{name}")

        save_result_bf(
            title=testcase['title'],
            pdc=logs['pdc'],
            ig=logs['ig'],
            path=path,
        )



# Estimate the interventional distribution for the hypothese m^hat_0, m^hat_1 with the observational data D_obs as described in 3.2


# pdf_H0 = m_obs_0.prob(D_Obs.dataY)
# pdf_H1 = m_obs_1.prob(D_Obs.dataY, x = D_Obs.dataX)
# prior_H0 = tf.reduce_mean(tf.cast(pdf_H0 > pdf_H1, dtype=dtype)).numpy()
