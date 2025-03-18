import os
from typing import List, Tuple, Dict, Any, Optional, Union

# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from utils.graphs import get_generic_graph
from sems.toy_sems import StationaryDependentSEM, StationaryIndependentSEM
from utils.tools import powerset, eager_replace
from utils.grids import get_interv_sampler, get_i_grids
from data_struct import hDict, Var, GraphObj, Node, esDict, IntervLog
from data_ops import DatasetObsDCBO, DSamplerObsDCBO, DatasetInv
from surrogates import Surrogate, SEMHat
from bo import ManualCausalEI, CausalEI
from bo import FixedCost
from models import BOModel
from copy import deepcopy
import tensorflow as tf
from utils.plots import save_result
from bo import CostBase
from sems import SEMBase

class DCBO:
    def __init__(

        self,
        nTrials: int,
        n_samples: int,
        dag: GraphObj,
        sem: SEMBase,
        exp_sets: List[Tuple[Var, ...]],
        interv_domain: Dict[Var, List[float]],
        dO_init: hDict,
        dO_eps: hDict,
        dtype: str = "float32",
    ) -> None:
        
        """
        Initializes the DCBO object.
        Parameters:
        -----------
        nTrials : int
            Number of trials to be performed.
        n_samples : int
            Number of intervention levels to be generated.
        dag : GraphObj
            Directed Acyclic Graph (DAG) object representing the causal structure.
        sem : SEMBase
            Structural Equation Model (SEM) object.
        exp_sets : List[Tuple[Var, ...]]
            List of experimental sets, each set is a tuple of variables.
        interv_domain : Dict[Var, List[float]]
            Dictionary mapping variables to their intervention domains.
        dO_init : hDict
            Initial observational data.
        dO_eps : hDict
            Perturbation to be added to the observational data.
        dtype : str, optional
            Data type for numerical operations, default is "float32".
        Returns:
        --------
        None
        """
        
        # Perform sanity checks on the input parameters
        self.sanity_check(
            nTrials,
            n_samples,
            dag,
            sem,
            variables=dag.variables,
            target_variable=dag.target_variable,
            exp_sets=exp_sets,
            interv_domain=interv_domain,
            dO_init=dO_init,
            dO_eps=dO_eps,
            dtype=dtype,
        )

        # Initialize class variables
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

        # Generate observational data
        self.genObsY, self.datasetO = self.obs(dO_init, dO_eps, dtype)
        
        # Initialize outcome values and BO models
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

        # Initialize intervention logger and datasets
        self.invLogger = IntervLog(exp_sets=exp_sets, nT=self.nT, nTrials=nTrials)
        self.invDX = esDict(exp_sets=exp_sets, nT=self.nT, nTrials=n_samples,)
        self.D_Inv = DatasetInv(exp_sets=exp_sets, nT=self.nT, nTrials=nTrials)
        self.dtype = dtype

    def sanity_check(
        self,
        nTrials: int,
        n_samples: int,
        dag: GraphObj,
        sem: SEMBase,
        variables: Union[List[Var], np.ndarray[Var]],
        target_variable: Var,
        exp_sets: List[Tuple[Var, ...]],
        interv_domain: Dict[Var, List[Union[float, int]]],
        dO_init: hDict,
        dO_eps: hDict,
        dtype: str,
    ) -> None:

        # Check if nTrials is a positive integer
        assert isinstance(nTrials, int) and nTrials > 0, "nTrials must be a positive integer"

        # Check if n_samples is a positive integer
        assert isinstance(n_samples, int) and n_samples > 0, "n_samples must be a positive integer"

        # Check if dag is an instance of GraphObj
        assert isinstance(dag, GraphObj), "dag must be an instance of GraphObj"

        # Check if sem is an instance of SEMBase
        assert isinstance(sem, SEMBase), "sem must be an instance of SEMBase"

        # Check if variables is a list of Var instances
        assert isinstance(variables, (List, np.ndarray)) and all(isinstance(v, Var) for v in variables), "variables must be a list of Var instances"

        # Check if target_variable is an instance of Var
        assert isinstance(target_variable, Var), "target_variable must be an instance of Var"

        # Check if exp_sets is a list of tuples of Var instances
        assert isinstance(exp_sets, (list, np.ndarray)) and all(isinstance(es, tuple) and all(isinstance(v, Var) for v in es) for es in exp_sets), "exp_sets must be a list of tuples of Var instances"

        # Check if interv_domain is a dictionary with Var keys and list of float values
        assert isinstance(interv_domain, dict) and all(isinstance(k, Var) and isinstance(v, list) and all(isinstance(i, (int, float)) for i in v) for k, v in interv_domain.items()), "interv_domain must be a dictionary with Var keys and list of float values"

        # Check if dO_init is an instance of hDict
        assert isinstance(dO_init, hDict), "dO_init must be an instance of hDict"

        # Check if dO_eps is an instance of hDict
        assert isinstance(dO_eps, hDict), "dO_eps must be an instance of hDict"
        
        # Check if dO_init and dO_eps contain all variables
        for var in variables:
            assert var in dO_init and var in dO_eps, "dO_init and dO_eps must contain all variables"

        # Check if acquisition cost is of type BaseCost
        assert isinstance(
            FixedCost(variables, 1), CostBase
        ), "Acquisition cost must be of type BaseCost"

        # Check if all variables in the exploration sets are covered
        exp_vars = set(var for es in exp_sets for var in es)
        assert exp_vars.issubset(
            set(variables)
        ), "All variables in the exploration sets must be covered"
        
        # check interv_domain keys are equal to variables in the exploration set
        assert set(interv_domain.keys()) == exp_vars, "interv_domain keys must be equal to variables in the exploration set"
        
        # Check the types of exp_sets
        assert all(
            isinstance(es, tuple) for es in exp_sets
        ), "Each exploration set must be a tuple"

        # Check if exp_sets are all subsets of variables \ target_variable
        assert all(
            set(es).issubset(set(variables) - {target_variable}) for es in exp_sets
        ), "Exploration sets must be subsets of variables excluding the target variable"
        
        # check the set of interv_domain keys equals to the exploration set
        assert set(interv_domain.keys()).issubset(
            set(variables)
        ), "interv_domain keys must be in variables"

        # Check if dtype is one of the allowed types
        assert isinstance(dtype, str) and dtype in ["float32", "float64"], "dtype must be one of the allowed types: float32, float64"
        assert set([v.name for v in variables]) == set(sem.static().keys()), "variables must be the same as the variables in sem"

    def obs(
        self, init_values: hDict, epsilon: hDict, dtype: str
    ) -> Tuple[DSamplerObsDCBO, DatasetObsDCBO]:
        """
        Generate observational data without intervention.
        Initial values and epsilon can be set.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to be generated.
        init_values : hDict
            Initial values for the variables.
        epsilon : hDict
            Perturbation to be added to the observational data.
        dtype : str
            Data type for numerical operations.
        
        """

        interv_levels = hDict(variables=self.variables, nT=self.nT)

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
                n_samples=epsilon.nTrials,
            ),
            dtype=dtype,
        )

        return genObsY, datasetO
    
    def get_gt(self) -> List[Tuple[Tuple[Var, ...], np.ndarray, float]]:
        """
        Get ground truth intervention levels and values.
        
        Returns:
        --------
        List[Tuple[Tuple[Var, ...], np.ndarray, float]]
            Ground truth data of the intervention - List of tuples containing the exploration set, intervention level, and the outcome value.
        """

        # Initialize intervention levels
        interv_levels = hDict(variables=self.variables, nT=self.nT, nTrials=1)
        n_samples = 100

        # Generate intervention grids
        igrid = get_i_grids(self.exp_sets, self.interv_domain, size_intervention_grid=5)

        result = []
        for t in range(self.nT):
            ce = {es: [] for es in self.exp_sets}
            for es in self.exp_sets:
                for level in igrid[es]:

                    # Deep copy of intervention levels
                    trial_levels = deepcopy(interv_levels)
                    for v, l in zip(es, level):
                        trial_levels[v][t] = l

                    # Generate new observational data
                    y_new = self.genObsY.sample(
                        initial_values=None,
                        interv_levels=trial_levels,
                        epsilon=None,
                        n_samples=n_samples,
                    )

                    # Monte Carlo sampling
                    mc = hDict(variables=self.variables, nT=1, nTrials=1)
                    for v in self.variables:
                        mc[v] = y_new[v][t].mean()

                    ce[es].append(mc[self.target_variable])

            # Find the minimum value for each exploration set
            loc = [
                (es, np.array(ce[es]).argmin(), np.min(ce[es])) for es in self.exp_sets
            ]
            m_es, idx, m_val = min(loc, key=lambda x: x[2])

            # Update intervention levels with the minimum value
            m_lvl = igrid[m_es][idx]
            for v, l in zip(m_es, m_lvl):
                interv_levels[v][t] = l

            result.append((m_es, m_lvl, m_val))

        return result

    def sc_run(self,acq_cost, init_tvalue):
        if init_tvalue is not None:
            assert isinstance(init_tvalue, (int, float)), "init_tvalue must be an integer or float"
        
        if acq_cost is not None:
            assert isinstance(acq_cost, CostBase), "acq_cost must be an instance of CostBase" 

    def run(
        self, acq_cost: Optional[CostBase], init_tvalue: Optional[Union[int, float]] = None
    ) -> IntervLog:
        """
        Run the DCBO algorithm.
        
        Parameters:
        -----------
        acq_cost : FixedCost
            Fixed cost for the acquisition function.
        init_tvalue : Optional[float]
            Initial value for the target variable.
            
        Returns:
        --------
        IntervLog
            Logger containing the experiment results.
        """
        
        self.sc_run(acq_cost, init_tvalue)
        self.surr = Surrogate(SEMHat(self.G, self.datasetO.dataY, dtype=dtype))

        # Initialize optimal intervention levels
        opt_ilvl = hDict(
            variables=self.variables,
            nT=self.nT,
            nTrials=1,
            default=lambda x, y, : np.array([[np.nan] * y] * x, dtype=self.dtype),
        )

        # Initialize trial intervention levels
        trial_ilvl = deepcopy(opt_ilvl)

        # Iterate over time steps
        for t in range(self.nT):
            # Iterate over trials
            for trial in range(self.nTrials):
                # Iterate over exploration sets
                for es in self.exp_sets:
                    # Create surrogate model
                    mean_f, variance_f = self.surr.create(
                        t=t,
                        interv_levels=opt_ilvl,
                        es=es,
                        target_var=self.target_variable,
                    )

                    # Initialize acquisition function
                    if self.bo_models[es][t, 0] is None:
                        acq = ManualCausalEI(
                            self.target_variable,
                            mean_f,
                            variance_f,
                            cmin=(
                                init_tvalue if t == 0 else self.invLogger.sol[t - 1][3]
                            ),
                        )
                    else:
                        acq = CausalEI(
                            bo_model=self.bo_models[es][t, 0],
                        )

                    # Sample intervention levels
                    self.invDX[es][t, :] = temp = self.interv_sampler[es].sample(
                        self.n_samples
                    )

                    # Evaluate improvements
                    improvements = acq.evaluate(
                        temp.astype(self.dtype)
                    ) 
                    if acq_cost is not None:
                        improvements /= acq_cost.evaluate(es, temp)

                    # Update intervention logger
                    self.invLogger.update(
                        t=t,
                        trial=trial,
                        i_set=es,
                        i_level=self.invDX[es][t, (idx := np.argmax(improvements))],
                        impv=improvements[idx],
                        y_values=None,
                    )

                # Get optimal intervention set and level for the trial
                _, trial_es, trial_lvl, _ = self.invLogger.opt_impv_trial[t, trial]

                # Update trial intervention levels
                for vid, var in enumerate(trial_es):
                    trial_ilvl[var] = (
                        opt_ilvl[var].numpy()
                        if isinstance(opt_ilvl[var], tf.Tensor)
                        else opt_ilvl[var]
                    )
                    trial_ilvl[var][t, 0] = trial_lvl[vid]

                # Generate new observational data
                y_new = self.genObsY.sample(
                    initial_values=None, interv_levels=trial_ilvl, n_samples=1
                )[self.target_variable][t][0]

                # Update intervention dataset
                self.D_Inv.update(t=t, es=trial_es, x=trial_lvl, y=y_new)

                # Update intervention logger with new outcome
                self.invLogger.update_y(t, trial, trial_es, y_new)

                # Update the BO model
                if self.bo_models[trial_es][t, 0] is None:
                    mean_f, variance_f = self.surr.create(
                        t=t,
                        interv_levels=trial_ilvl,
                        es=trial_es,
                        target_var=self.target_variable,
                    )

                    self.bo_models[trial_es][t, 0] = BOModel(
                        mean_f,
                        variance_f,
                        dtype = self.dtype,
                    )

                # Fit the BO model with new data
                dataIX, dataIY = self.D_Inv.get_tf(trial_es, t)
                self.bo_models[trial_es][t, 0].fit(
                    dataIX, dataIY, n_restart=10, verbose=False
                )
                print("time: ", t, "trial: ", trial, "es: ", trial_es, "y: ", y_new)

            # Update optimal intervention levels
            _, best_es, best_lvl, _ = self.invLogger.sol[t]
            for vid, var in enumerate(best_es):
                opt_ilvl[var] = eager_replace(
                    opt_ilvl[var], best_lvl[vid, None], t, axis=0, dtype=self.dtype
                )

        return self.invLogger


if __name__ == "__main__":
    # Define the number of trials, samples, and observational data points
    nTrials = 10
    n_samples = 25
    n_obs = 100
    dtype = "float64"
    nT = 3

    # Define the variables and target variable
    variables = [Var("X"), Var("Z"), Var("Y")]
    target_variable = Var("Y")

    # Generate all possible experimental sets
    exp_sets = list(powerset([Var("X"), Var("Z")]))

    # Initialize the initial values dictionary
    initial_values = hDict(variables=variables)

    # Function to generate epsilon values for the variables
    def agn_eps(dd: Dict[Var, Tuple[float, float]]) -> hDict:
        x = hDict(
            variables=variables,
            nT=nT,
            nTrials=n_obs,
        )
        for var, (mean, std) in dd.items():
            x[var] = np.random.normal(mean, std, (nT, n_obs))
        return x

    # Define the test cases with different SEMs and epsilon values
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

    # Iterate over each test case
    for testcase in testing:

        # Create the graph object
        dag = get_generic_graph(
            stop_time=nT,
            topology=testcase["topology"],
            nodes=variables,
            target_node=target_variable,
        )
        G = GraphObj(graph=dag, nT=nT, target_var=target_variable)

        # Initialize the DCBO object
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

        # Run the DCBO algorithm
        logger = dcbo.run(FixedCost(variables, 1), init_tvalue=10)
        print(logger.sol)
        print(gt := dcbo.get_gt())

        # Save the results
        name = testcase["name"]
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "results", f"{name}"
        )
        save_result(
            testcase["title"], [x[-1] for x in gt], logger.opt_y_trial[:, :, 3], path
        )

        del dcbo
