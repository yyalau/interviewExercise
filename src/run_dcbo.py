import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import numpy as np
from utils.graphs import get_generic_graph
from sems.toy_sems import StationaryDependentSEM, StationaryIndependentSEM
from sems import SEMHat
from utils.tools import powerset, eager_replace
from utils.grids import get_interv_sampler, get_i_grids
from data_struct import hDict, Var, GraphObj, Node, esDict, IntervLog
from data_ops import DatasetObsDCBO, DSamplerObsDCBO, DatasetInv
from surrogates import PriorEmit, PriorTrans, Surrogate
from bo import ManualCausalEI, CausalEI, BOModel
from bo import FixedCost
from copy import deepcopy
import tensorflow as tf
from utils.plots import save_result


class DCBO:
    def __init__(
        self,
        nTrials,
        n_samples,
        dag,
        sem,
        exp_sets,
        interv_domain,
        dO_init,
        dO_eps,
        dtype="float32",
    ):

        self.sanity_check()

        self.nTrials = nTrials
        self.n_samples = n_samples
        self.sem = sem
        self.exp_sets = exp_sets
        self.interv_domain = interv_domain
        self.interv_sampler = get_interv_sampler(exp_sets, interv_domain)

        self.G = dag
        self.variables = dag.variables
        self.target_variable = dag.target_variable
        self.nT = dag.nT

        self.genObsY, self.datasetO = self.obs(self.n_samples, dO_init, dO_eps, dtype)
        self.surr = self.get_surr(dag, self.datasetO)

        self.outcome_values = hDict(
            variables=self.variables,
            nT=self.nT,
            nTrials=self.nTrials,
        )

        self.bo_models = hDict(
            variables=self.exp_sets,
            nT=self.nT,
            nTrials=1,
        )
        #
        self.invLogger = IntervLog(exp_sets=exp_sets, nT=self.nT, nTrials=nTrials)
        self.invDX = esDict(exp_sets=exp_sets, nT=self.nT, nTrials=n_samples)
        self.D_Inv = DatasetInv(exp_sets=exp_sets, nT=nT, nTrials=nTrials)

    def sanity_check(self):
        """
        assertions
        1. assert type of acquisition cost (BaseCost), and all variables in the exploration sets are covered
        2. check the types of exp_sets
        3. exp_sets all subsets of variables \ target_variable
        4. assert the dtype in float32, float64, double (search for possible types)
        """
        pass

    def obs(self, n_samples, init_values, epsilon, dtype):
        """
        1. no intervention
        2. initial values and epsilon can be set
        """

        interv_levels = hDict(variables=variables, nT=nT)

        genObsY = DSamplerObsDCBO(
            sem=self.sem,
            nT=self.nT,
            variables=self.variables,
            dtype=dtype,
        )

        datasetO = DatasetObsDCBO(
            initial_values=init_values,
            interv_levels=interv_levels,
            epsilon=epsilon,
            dataY=genObsY.sample(
                initial_values=init_values,
                interv_levels=interv_levels,
                epsilon=epsilon,
                n_samples=n_samples,
            ),
            dtype=dtype,
        )

        return genObsY, datasetO

    def get_gt(self):

        interv_levels = hDict(variables=self.variables, nT=self.nT, nTrials=1)
        n_samples = 100

        igrid = get_i_grids(self.exp_sets, self.interv_domain, size_intervention_grid=5)

        result = []
        for t in range(self.nT):
            ce = {es: [] for es in self.exp_sets}
            for es in self.exp_sets:
                for level in igrid[es]:

                    trial_levels = deepcopy(interv_levels)
                    for v, l in zip(es, level):
                        trial_levels[v][t] = l

                    y_new = self.genObsY.sample(
                        initial_values=None,
                        interv_levels=trial_levels,
                        epsilon=None,
                        n_samples=n_samples,
                    )

                    # monte carlo sampling
                    mc = hDict(variables=self.variables, nT=1, nTrials=1)
                    for v in self.variables:
                        mc[v] = y_new[v][t].mean()

                    ce[es].append(mc[self.target_variable])

            loc = [
                (es, np.array(ce[es]).argmin(), np.min(ce[es])) for es in self.exp_sets
            ]
            m_es, idx, m_val = min(loc, key=lambda x: x[2])

            m_lvl = igrid[m_es][idx]
            for v, l in zip(m_es, m_lvl):
                interv_levels[v][t] = l

            result.append((m_es, m_lvl, m_val))

        return result

    def get_surr(self, G, datasetO):

        def fitted_prior(prior, G, dataY):
            p = prior(G)
            p.fit(dataY)
            return p

        prior_emit = fitted_prior(PriorEmit, G, datasetO.dataY)
        prior_trans = fitted_prior(PriorTrans, G, datasetO.dataY)

        return Surrogate(SEMHat(self.G, prior_emit, prior_trans))
    

    def run(self, acq_cost, init_tvalue=None):

        opt_ilvl = hDict(
            variables=self.variables,
            nT=self.nT,
            nTrials=1,
            default=lambda x, y: np.array([[np.nan] * y] * x, dtype=dtype),
        )

        trial_ilvl = deepcopy(opt_ilvl)

        for t in range(self.nT):
            for trial in range(self.nTrials):
                for es in self.exp_sets:
                    mean_f, variance_f = self.surr.create(
                        t=t, interv_levels=opt_ilvl, es=es, target_var=target_variable
                    )

                    if self.bo_models[es][t, 0] is None:
                        acq = ManualCausalEI(
                            self.target_variable,
                            mean_f,
                            variance_f,
                            cmin=init_tvalue if t == 0 else self.invLogger.sol[t-1][3],
                        )
                    else:
                        acq = CausalEI(
                            bo_model=self.bo_models[es][t, 0],
                        )

                    self.invDX[es][t, :] = temp = self.interv_sampler[es].sample(
                        self.n_samples
                    )

                    improvements = acq.evaluate(temp.astype(dtype)) / acq_cost.evaluate(
                        es, temp
                    )

                    self.invLogger.update(
                        t=t,
                        trial=trial,
                        i_set=es,
                        i_level=self.invDX[es][t, (idx := np.argmax(improvements))],
                        impv=improvements[idx],
                        y_values=None,
                    )

                _, trial_es, trial_lvl, _ = self.invLogger.opt_impv_trial[t, trial]

                for vid, var in enumerate(trial_es):
                    trial_ilvl[var] = (
                        opt_ilvl[var].numpy()
                        if isinstance(opt_ilvl[var], tf.Tensor)
                        else opt_ilvl[var]
                    )
                    trial_ilvl[var][t, 0] = trial_lvl[vid]

                y_new = self.genObsY.sample(
                    initial_values=None, interv_levels=trial_ilvl, n_samples=1
                )[target_variable][t][0]

                self.D_Inv.update(t=t, es=trial_es, x=trial_lvl, y=y_new)

                self.invLogger.update_y(t, trial, trial_es, y_new)

                # update the BO model
                if self.bo_models[trial_es][t, 0] is None:
                    mean_f, variance_f = self.surr.create(
                        t=t,
                        interv_levels=trial_ilvl,
                        es=trial_es,
                        target_var=target_variable,
                    )

                    self.bo_models[trial_es][t, 0] = BOModel(
                        es,
                        target_variable,
                        mean_f,
                        variance_f,
                    )

                dataIX, dataIY = self.D_Inv.get_tf(trial_es, t)
                self.bo_models[trial_es][t, 0].fit(
                    dataIX, dataIY, n_restart=10, verbose=False
                )
                print("time: ", t, "trial: ", trial, "es: ", trial_es, "y: ", y_new)

            # update opt_ilvl
            _, best_es, best_lvl, _ = self.invLogger.sol[t]
            for vid, var in enumerate(best_es):
                opt_ilvl[var] = eager_replace(
                    opt_ilvl[var], best_lvl[vid, None], t, axis=0, dtype=dtype
                )

        return self.invLogger


if __name__ == "__main__":
    nTrials = 10
    n_samples = 25
    dtype = "float32"
    nT = 3
    variables = [Var("X"), Var("Z"), Var("Y")]
    target_variable = Var("Y")
    exp_sets = list(powerset([Var("X"), Var("Z")]))
    initial_values = hDict(variables=variables)

    def agn_eps(dd):
        x = hDict(
            variables=variables,
            nT=nT,
            nTrials=n_samples,
        )
        for var, (mean, std) in dd.items():
            x[var] = np.random.normal(mean, std, (nT, n_samples))
        return x

    testing = [
        {
            "name": "noisy",
            "sem": StationaryDependentSEM(),
            "eps": {
                Var("X"): (2, 4),
                Var("Z"): (2, 4),
                Var("Y"): (0, 1),
            },
            "domain": {Var("X"): [-5, 5], Var("Z"): [-5, 20]},
            "topology": "dependent",
            "title": "Noisy",
        },
        {
            "name": "stat",
            "sem": StationaryDependentSEM(),
            "eps": {
                Var("X"): (0, 1),
                Var("Z"): (0, 1),
                Var("Y"): (0, 1),
            },
            "domain": {Var("X"): [-5, 5], Var("Z"): [-5, 20]},
            "topology": "dependent",
            "title": "Stationary",
        },
        {
            "name": "ind",
            "sem": StationaryIndependentSEM(),
            "eps": {
                Var("X"): (0, 1),
                Var("Z"): (0, 1),
                Var("Y"): (0, 1),
            },
            "domain": {Var("X"): [-5, 5], Var("Z"): [-5, 20]},
            "topology": "independent",
            "title": "Independent",
        },
    ]

    for testcase in testing:

        # create graph
        dag = get_generic_graph(
            start_time=0,
            stop_time=nT,
            topology=testcase["topology"],
            nodes=variables,
            target_node=target_variable,
        )
        G = GraphObj(graph=dag, nT=nT, target_var=target_variable)

        dcbo = DCBO(
            nTrials,
            n_samples,
            G,
            testcase["sem"],
            exp_sets,
            testcase["domain"],
            initial_values,
            agn_eps(testcase["eps"]),
            dtype,
        )
        acq_cost = FixedCost(variables, 1)
        logger = dcbo.run(acq_cost, init_tvalue=10)
        print(logger.sol)
        print(gt := dcbo.get_gt())
        
        name = testcase["name"]
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)),  "results", f"{name}")
        save_result(testcase["title"], [x[-1] for x in gt], logger.opt_y_trial[:, :, 3], path)
        
        del dcbo
