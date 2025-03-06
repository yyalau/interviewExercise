    

from graphviz import Source
from numpy import repeat
from itertools import cycle, chain
from networkx import MultiDiGraph
from typing import List, Union
import pygraphviz as pgv
from networkx.drawing import nx_agraph



if __name__ == "__main__":
    import sys,os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data_struct import Node, Var


def get_generic_graph(
    start_time: int,
    stop_time: int,
    topology: str,
    nodes: List[str],
    target_node: str = None,
) -> MultiDiGraph:
    
    if topology == "independent":
        assert target_node is not None
        assert isinstance(target_node, Var)

    spatial_edges = []

    if topology == "independent": 
        for t in range(start_time, stop_time):
            spatial_edges.extend([f"{node}_{t} -> {target_node}_{t};" for node in nodes if node != target_node])

    elif topology == "dependent":
        for t in range(start_time, stop_time):
            spatial_edges.extend([f"{node1}_{t} -> {node2}_{t};" for node1, node2 in zip(nodes[:-1], nodes[1:])])            
    else:
        raise ValueError("Topology not recognized")    
    
    ranking = "".join(["{{ rank=same; {} }} ".format(
                " ".join(["{}_{}".format(node, t) for node in nodes])
            ) for t in range(start_time, stop_time) ])

    spatial_edges = "".join(spatial_edges)    
    temporal_edges = "".join([ f"{node}_{t} -> {node}_{t+1};" for t in range(start_time, stop_time - 1) for node in nodes ])
    graph = "digraph {{ rankdir=LR; {} {} {} }}".format(
        spatial_edges, temporal_edges, ranking
    )
    
    return nx_agraph.from_agraph(pgv.AGraph(graph))

def vis_graph(graph: str, filename: str) -> None:
    """
    Visualize the graph using graphviz.

    Parameters
    ----------
    graph : str
        DOT format of the graph
    filename : str
        Name of the file to save the graph (WITHOUT extension)
    """
    # TODO: the case with filename.extension
    
    # remove the file extension if it exists
    if "." in filename:
        file = filename.split(".")
        filename, ext = file 
    else:
        ext = "png"

    src = Source(graph)
    src.render(filename, format=ext)

if __name__ == "__main__":
    nT = 4
    G = get_generic_graph(
        start_time=0,
        stop_time=nT,
        topology="dependent",
        nodes=[Var("X"), Var("Z"), Var("Y"), Var("W")],
        target_node=Var("Y"),
    )
