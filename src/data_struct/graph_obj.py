from .node import Node, Var
import numpy as np
from networkx.algorithms.dag import topological_sort


class GraphObj:
    def __init__(self, graph, nT, target_var):
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

    def __str__(self):
        return f"Graph Object: {self.dag} \nnT: {self.nT} \nnVar: {self.nVar} \nnodes: {self.nodes} \nvariables: {self.variables}"
