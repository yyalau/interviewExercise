import os
from typing import List, Union, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from sems.syn import X2Y, Y2X, Confounder
from data_ops import DatasetBF, DSamplerObsBF
from data_struct import hDict, Var
import numpy as np
from models import MoG, GLMTanh
from utils.tools import get_probH0, update_posteriorH0
from utils.metrics import PDC, InfoGain
from utils.plots import save_result_bf
from sems  import SEMBase

class HypothesisTesting:
    def __init__(
        self, sem: SEMBase, variables: List[Var], varX: Var, varY: Var, n_pdc: int, dO_eps: hDict, criterion: Union[PDC, InfoGain], m_obs_0: MoG, m_obs_1: MoG, dtype: str
    ):
        """
        Parameters:
        -----------
        sem : SEMBase
            The structural equation model used for data generation.
        variables : List[Var]
            A list of variables present in the SEM.
        varX : Var
            The variable of interest in hypothesis H1 (to test for causality with varY).
        varY : Var
            The target variable in hypotheses H0 and H1.
        n_pdc : int
            Number of samples to draw from the distribution m_obs_0 and m_obs_1 to calculate the criterion (P_DC or InfoGain).
        m_obs_0 : MoG
            The model for the observed data under hypothesis H0.
        m_obs_1 : MoG
            The model for the observed data under hypothesis H1.
        dO_eps : hDict
            The noise for the observed data.
        criterion : Union[PDC, InfoGain]
            The criterion to use for the hypothesis testing.
        dtype : str
            The data type of the values. One of "float32" or "float64".
        """
        self.sanity_check(sem, variables, varX, varY, n_pdc, dO_eps, criterion, m_obs_0, m_obs_1, dtype)
        
        self.sem = sem
        self.variables = variables
        self.varX = varX
        self.varY = varY
        self.n_pdc = n_pdc
        self.m_obs_0 = m_obs_0      
        self.m_obs_1 = m_obs_1
        self.genObsY, self.D_Obs = self.obs(sem, variables, varX, varY, dO_eps, dtype)
        self.D_Int = DatasetBF(dtype=dtype)
        self.criterion = criterion
        self.dtype = dtype

    def sanity_check(self, sem, variables, varX, varY, n_pdc, dO_eps, criterion, m_obs_0, m_obs_1, dtype):

        
        assert isinstance(sem, SEMBase), "sem must be an instance of SEMBase"
        assert isinstance(variables, list) and all(isinstance(var, Var) for var in variables), "variables must be a list of Var instances"
        assert isinstance(varX, Var), "varX must be an instance of Var"
        assert isinstance(varY, Var), "varY must be an instance of Var"
        assert isinstance(n_pdc, int) and n_pdc > 0, "n_pdc must be a positive integer"
        assert isinstance(dO_eps, hDict), "dO_eps must be an instance of hDict"
        assert isinstance(criterion, (PDC, InfoGain)), "criterion must be an instance of PDC or InfoGain"
        assert isinstance(m_obs_0, MoG), "m_obs_0 must be an instance of MoG"
        assert isinstance(m_obs_1, MoG), "m_obs_1 must be an instance of MoG"
        assert isinstance(dtype, str), "dtype must be a string"
        assert dtype in ["float32", "float64"], "dtype must be either 'float32' or 'float64'"
        
        
        assert varX in variables, "varX must be in variables"
        assert varY in variables, "varY must be in variables"
        assert varX != varY, "varX must not be equal to varY"
        assert set([v.name for v in variables]) == set(sem.static().keys()), "variables must be the same as the variables in sem"
        
    def obs(self, sem: SEMBase, variables: List[Var], varX: Var, varY: Var, epsilon: hDict, dtype: str) -> Tuple[DSamplerObsBF, DatasetBF]:
        '''
        Generates the observed data.
        Parameters:
        -----------
        sem : SEMBase
            The structural equation model used for data generation.
        variables : List[Var]
            A list of variables present in the SEM.
        n_obs : int
            Number of observations to generate.
        epsilon : hDict
            The noise for the observed data.
        dtype : str
            The data type of the values. One of "float32" or "float64".
        Returns:
        --------
        genObsY : DSamplerObsBF
            The data sampler for the observed data.
        D_Obs : DatasetBF
            The observed dataset.
        '''
        
        genObsY = DSamplerObsBF(sem=sem, variables=variables, dtype=dtype)

        obsY = genObsY.sample(epsilon=epsilon, n_samples=epsilon.nTrials)
        D_Obs = DatasetBF(
            dataX=epsilon[varX][0].reshape(-1, 1),
            dataY=obsY[varY][0].reshape(-1, 1),
            dtype=dtype,
        )
        return genObsY, D_Obs
    
    def run(self, n_trials: int) -> List[float]:
        '''
        Runs the hypothesis testing.
        Parameters:
        -----------
        n_trials : int
            Number of trials to run.
        Returns:
        --------
        logs : List[float]
            The logs of the criterion values.
        '''
        
        # Ensure the number of trials is a positive integer
        assert isinstance(n_trials, int) and n_trials > 0, "n_trials must be a positive integer"
        
        logs = []
        
        # Fit the models for the observed data under hypotheses H0 and H1
        self.m_obs_0.fit(self.D_Obs.dataX, self.D_Obs.dataY, verbose=False)
        self.m_obs_1.fit(self.D_Obs.dataX, self.D_Obs.dataY, verbose=False)

        # Calculate the prior probability of H0
        prior_H0 = get_probH0(
            self.D_Obs.dataX, self.D_Obs.dataY, self.m_obs_0, self.m_obs_1, dtype=self.dtype
        )
        posterior_H0 = 0.5  # Initialize the posterior probability of H0
        print("P(H_0) = ", prior_H0, "; P(H_1) = ", 1 - prior_H0)

        for trial in range(n_trials):
            # Fit the criterion to the data
            result = self.criterion.fit(
                prior_H0,
                self.D_Int,
                self.m_obs_0,
                self.m_obs_1,
                n_samples=self.n_pdc,
                n_restart=10,
                verbose=False,
            )
            x_opt = self.criterion.get_xopt()  # Get the optimal intervention value

            # Create a temporary hDict for the intervention
            temp = hDict(
                variables=self.variables,
                nT=1,
                nTrials=1,
                default=lambda x, y: np.zeros((x, y)).astype(self.dtype),
            )
            temp[self.varX] = x_opt  # Set the intervention value for varX
            temp[self.varY] = np.zeros((1, 1))  # Initialize varY with zeros

            # Sample new data based on the intervention
            y_new = self.genObsY.sample(epsilon=temp, n_samples=1)

            # Update the intervention dataset with the new sample
            self.D_Int.update(y_new[self.varX], y_new[self.varY])
            # Update the posterior probability of H0
            posterior_H0 = update_posteriorH0(
                self.D_Int, self.m_obs_0, self.m_obs_1, prior_H0
            )

            # Print the results of the current trial
            print(
                "trial: ",
                trial,
                "; P(H_0 | D_int) = ",
                posterior_H0,
                "; P(H_1 | D_int) = ",
                1 - posterior_H0,
                "; result = ", result.numpy(),
            )
            logs.append(result.numpy()*-1)  # Append the result to the logs

        return logs

if __name__ == "__main__":
    # Define the variables of interest
    varX = Var("X")
    varY = Var("Y")
    dtype = "float32"    
    n_obs = 3000  # Number of observations

    # Define the test cases for hypothesis testing
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
    n_pdc = 200  # Number of samples for criterion calculation
    n_dist = 2  # Number of distributions for MoG model
    
    # Define the parameters for the PDC metric
    beta = 0.2
    k0 = 10
    k1 = 1.0 / k0

    # Define the metrics to be used for hypothesis testing
    metrics = {
        "pdc": PDC(beta, k0, k1, dtype=dtype),
        "ig": InfoGain(dtype=dtype),
    }

    # Loop through each test case
    for testcase in testing:
        logs = {}
        # Loop through each metric
        for m_name, m_f in metrics.items():
        
            # Generate noise for the observed data
            epsilon = hDict(
                variables=testcase['variables'],
                nT=1,
                nTrials=n_obs,
                default=lambda x, y,: np.random.randn(x, y).astype(dtype),
            )

            # Define the models for the observed data under H0 and H1
            m_obs_0 = MoG(n_dist=n_dist, dtype=dtype)
            m_obs_1 = MoG(n_dist=n_dist, dtype=dtype, link_fn=GLMTanh(dtype=dtype))

            n_trials = 10  # Number of trials for intervention

            # Initialize the HypothesisTesting class
            ht = HypothesisTesting(
                testcase['sem'], testcase['variables'], varX, varY, n_pdc, epsilon, m_f, m_obs_0, m_obs_1, dtype
            )
            # Run the hypothesis testing and store the logs
            logs[m_name] = ht.run(n_trials)
            
        # Define the path to save the results
        name = testcase['name']
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)),  "results", f"{name}")

        # Save the results
        save_result_bf(
            title=testcase['title'],
            pdc=logs['pdc'],
            ig=logs['ig'],
            path=path,
        )
