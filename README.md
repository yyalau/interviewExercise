# Interview Exercise

Name: Lau, Ying Yee Ava

Topic: **[Question 3]** Dynamic causal Bayesian optimization and optimal intervention.

## Submission

- [Final Report](reports/final_report.pdf)  

- Code: Part 1 and part 2 share the same [code base](./src)
    - Part 1: [DCBO](./src/run_dcbo.py) 
    - Part 2: [Bayesian Intervention Optimization for Causal Discovery](./src/run_bf.py)

- [Testing](./tests) contains the test cases for the code.

## Getting Started

This code is written in **Python 3.8.0** on **Linux (AlmaLinux 9.5)**. To install the required packages, run the following commands:
```bash
# create a virtual environment
python -m venv venv

# activate the virtual environment
source venv/bin/activate

# install the required packages
pip install -r requirements.txt
```
Before running the code, create a directory `results` in the root directory, where the output logs will be stored.
```bash
mkdir results
```

To run DCBO (Part 1), 
```bash
python src/run_dcbo.py
```
To run the optimal intervention (Part 2), run the following command in the terminal
```bash
python src/run_bf.py
```
To run the tests, run the following command in the terminal
```bash
python -m pytest 
```

## Basic Program Structure (`src`)
- `bo`: Contains utility functions that support the Bayesian optimization process.
    - `acquisition`: Contains the acquisition functions 
        - `base.py`: Contains the base class (`EIBase`) for acquisition functions
        - `ei.py`: Contains the Expected Improvement acquisition function when the BO model is available
        - `manual_ei.py`: Contains the Expected Improvement acquisition function when the BO model has not been instantiated
    - `cost.py`: Contains the cost functions
- `data_opts`: Contains dataset objects for data feeding and samplers for sampling operations
    - `dataset`: Contains the observation and intervention dataset objects
        - `bf.py`: Intervention dataset for the optimal intervention (Part 2)
        - `intervention.py`: Intervention dataset for the DCBO (Part 1)
        - `observation.py`: Observation dataset for the DCBO (Part 1)
    - `samplers`: Contains the samplers for sampling observational data based on the SEMs
        - `observation.py`: Contains the base class `DSamplerObsBase`. The `DSamplerObsDCBO` and `DSamplerObsBF` are the subclasses for sampling observational data for DCBO (Part 1) and optimal intervention (Part 2), respectively.
    - `base.py`: Contains the base class for the dataset objects `DatasetBase` and the sampler objects `DataSamplerBase`
- `data_struct`: Contains self-defined data structures for the ease of data manipulation
    - `graph_obj.py`: Data type for the causal DAG (`GraphObj`)
    - `interv.py`: Contains `IntervLog` for logging for the improvements, intervention sets, levels and outcomes
    - `newdict.py`: Contains the `hDict` class for storing graph data. The keys are ordered by graph variables, time, and the number of trials for the intervention. Contains the `esDict` class for storing data for the exploration sets. The keys are ordered by the exploration sets, time, and the number of trials for the intervention. Both classes are inherited from the `OrderedDict` class and used in both DCBO (Part 1) and optimal intervention (Part 2).
    - `node.py`: Contains the `Var` and `Node` class for the variable object and node object in the causal DAG respectively. The `Node` class contains the `Var` object.
- `models`: Contains the model classes, written in TensorFlow 2.0 and TensorFlow Probability 
    - `base.py`: Defines the `NLLBase` class for initializing and training models optimized using the negative log-likelihood loss.
    - `bo_model.py`: Implements the `BOModel` class for the Bayesian optimization model.
    - `causalrbf.py`: Implements the `CausalRBF` class, which defines a custom variance function for the RBF kernel. The `GaussianRBF` and `GammaRBF` classes are subclasses of the `CausalRBF` class, which customizes the variance functions for the Gaussian and Gamma kernels respectively.
    - `glm.py`: Implements the `GLMTanh` class for the Generalized Linear Model with the `tanh` link function.
    - `gps.py`: Implements the `GPRegression` class for the Gaussian Process model.
    - `kde.py`: Implements the `KernelDensity` class for the Kernel Density Estimation model.
    - `mog.py`: Implements the `MoG` class for the Mixture of Gaussians model. Also contains a `ExpNorm` class, which is a Tensorflow Probability Bijector for transforming the vector to be positive and ensuring the sum of the vector is 1.
- `sems`: Contains the structural equation models for the causal discovery process
    - `base.py`: Contains the base class `SEMBase` for the structural equation models
    - `real_sems.py`: From the original DCBO repository. Contains the real-world SEMs for the causal discovery process.
    - `semhat.py`: Implements the `SEMHat` class, which constructs Gaussian Process and Kernel Density Estimation models for each node in the causal graph. Specifically, it takes in `PriorEmit` and `PriorTrans` objects that provides fitted distributions for the emission and transition probabilities of the DAG nodes respectively.
    - `syn.py`: Contains the synthetic SEMs used in Part 2
    - `toy_sems.py`: Contains the synthetic SEMs used in Part 1
- `surrogates`: Contains the surrogate models for the Bayesian optimization process
    - `prior`: Creates functions to fit the prior distribution for the Bayesian optimization process
        - `emit.py`: Contains the `PriorEmit` class for fitting the prior distribution of emission probabilities of the DAG nodes
        - `trans.py`: Contains the `PriorTrans` class for fitting the prior distribution of transition probabilities of the DAG nodes
    - `base.py`: Contains the base class `PriorBase` for fitting the priors
    - `surr.py`: Contains the class `Surrogate`, which accepts `SEMHat` objects. It creates the surrogate mean and variance functions for the Bayesian optimization process
- `utils`: Contains utility functions that support the main program
    - `graphs.py`: Provides functions for creating causal DAGs.
        - `grids.py`: Includes functions for generating grids for intervention levels used in Part 1. These grids are utilized in data sampling and ground truth data generation.
        - `metrics.py`: Includes `BayesFactor` class for calculating the Bayes Factor. `PDC` metric for the optimization objective and `InfoGain` class for calculating the information gain. These metrics are used in the optimal intervention (Part 2). 
        - `plots.py`: Includes functions for saving plots and numerical results.
        - `tools.py`: Includes utility functions for the main program.
- `run_dcbo.py`: The main program for DCBO (Part 1)
- `run_bf.py`: The main program for the optimal intervention (Part 2)