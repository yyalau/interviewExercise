from typing import List, Any
from .node import Node, Var
import numpy as np
from networkx.algorithms.dag import topological_sort
from networkx import DiGraph

class GraphObj:
    def __init__(self, graph: DiGraph, nT: int, target_var: Var) -> None:
        '''
        Initializes the GraphObj object with the given graph, number of time steps, and target variable.
            
        Parameters:
        -----------
        graph : DiGraph
            The directed acyclic graph.
        nT : int
            The number of time steps.
        target_var : Var
            The target variable.
        '''
        self.dag = graph
        self.nT = nT
        self.target_variable = target_var
        assert (
            graph.number_of_nodes() % nT == 0
        ), "Number of nodes must be divisible by number of time steps."

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

    def __str__(self) -> str:
        return f"Graph Object: {self.dag} \nnT: {self.nT} \nnVar: {self.nVar} \nnodes: {self.nodes} \nvariables: {self.variables}"
