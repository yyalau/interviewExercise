from typing import List, Any
from .node import Node, Var
import numpy as np
from networkx.algorithms.dag import topological_sort
from networkx import DiGraph
import networkx as nx

class GraphObj:
    def __init__(self, graph: DiGraph, nT: int, target_var: Var) -> None:
        '''
        Initializes the GraphObj object with the given graph, number of time steps, and target variable. 
                    
        Parameters:
        -----------
        graph : DiGraph
            The directed acyclic graph.
            Requirements for the 'graph':
            - The graph must be a directed acyclic graph.
            - The nodes must be named as <variable>_<time_step>.
            - The set of variables must be the same for all time steps.
            - The target variable must be in the graph.
        nT : int
            The number of time steps.
        target_var : Var
            The target variable.
        '''
        
        self.sc(graph, nT, target_var)
        
        self.dag = graph
        self.nT = nT
        self.target_variable = target_var

        self.nVar = graph.number_of_nodes() // nT
        self.nodes = np.array(
            sorted(
                [Node(*n.split("_")) for n in topological_sort(graph)],
                key=lambda x: x.t,
            )
        )
        
        uvars = [v for v in set([n.name for n in self.nodes])]
        uvars.remove(target_var)
        uvars.append(target_var)
        self.variables = np.array(uvars)
        
    def sc(self, graph, nT, target_var):
        
        # check nT is an integer greater than 0
        assert nT > 0 and isinstance(nT, int), "The number of time steps must be an integer greater than 0."
        
        # check target_var is a Var object
        assert isinstance(target_var, Var), "The target variable must be a Var object."
        
        # check target_var is in the graph
        assert target_var.name in [node.split("_")[0] for node in graph.nodes], f"The target variable {target_var} is not in the graph"
        
        # Check if the graph is a DAG
        assert nx.is_directed_acyclic_graph(graph), "The graph must be a directed acyclic graph."

        for node in graph.nodes:
            # check if the nodes are made up of <var>_<time_step>
            hh = node.split("_")
            assert len(hh) == 2 and hh[1].isdigit(), "Nodes must be named as <variable>_<time_step>."
            
            # check if the time step is less than nT
            assert hh[1] < str(nT), "Time step must be less than the number of time steps."

        # check or all time steps, the set of variables is the same
        mm = [0 for t in range(nT)]
        for t in range(nT):
            mm[t] = set([n.split("_")[0] for n in graph.nodes if n.split("_")[1] == str(t)])    
        assert all([ set(m) == mm[0] for m in mm]), "The set of variables must be the same for all time steps."
            
        assert (
            graph.number_of_nodes() % nT == 0
        ), "Number of nodes must be divisible by number of time steps."
        
        assert (
            graph.number_of_nodes() // nT == len(set([n.split("_")[0] for n in graph.nodes]))
        ), "Number of unique variables must be equal to the number of variables."
        
        # Check if the target variable is in the graph
        assert target_var in [node.split("_")[0] for node in graph.nodes], f"The target variable {target_var} is not in the graph"

    def __str__(self) -> str:
        return f"Graph Object: {self.dag} \nnT: {self.nT} \nnVar: {self.nVar} \nnodes: {self.nodes} \nvariables: {self.variables}"
