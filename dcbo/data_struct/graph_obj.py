from .node import Node, Var
import numpy as np

class GraphObj:
    def __init__(self, graph, nT):
        self.dag = graph
        self.nT = nT
        assert graph.number_of_nodes() % nT == 0, "Number of nodes must be divisible by number of time steps."
        
        self.nVar = graph.number_of_nodes() // nT
        self.nodes = np.array([Node(*n.split('_')) for n in graph.nodes()])
        
        uvars = set([n.name for n in self.nodes])
        self.variables = np.array([v for v in uvars])
    


    def __str__(self):
        return f"Graph Object: {self.dag} \nnT: {self.nT} \nnVar: {self.nVar} \nnodes: {self.nodes} \nvariables: {self.variables}"
