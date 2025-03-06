# Interview Exercise

Name: Lau, Ying Yee Ava

Topic: **[Question 3]** Dynamic causal Bayesian optimization and optimal intervention.

## Submission

- Report: [Final](reports/final_report.pdf)  

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
- To run DCBO (Part 1), 
    ```bash
    python src/run_dcbo.py
    ```
- To run the optimal intervention (Part 2), run the following command in the terminal
    ```bash
    python src/run_bf.py
    ```
- To run the tests, run the following command in the terminal
    ```bash
    python -m pytest 
    ```

## Basic Program Structure (`src`)
- TODO