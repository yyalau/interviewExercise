from graphviz import Source
from numpy import repeat
from itertools import cycle, chain
from networkx import MultiDiGraph
from typing import List, Union
import pygraphviz as pgv
from networkx.drawing import nx_agraph
from data_struct import Node


def get_generic_graph(
    start_time: int,
    stop_time: int,
    topology: str,
    nodes: List[str],
    target_node: str = None,
) -> MultiDiGraph:
    """
    Generic temporal Bayesian network with two types of connections.

    Parameters
    ----------
    start : int
        Index of first time-step
    stop : int
        Index of the last time-step
    topology: str, optional
        Choice of independent and dependent causal topology
    nodes: list
        List containing the nodes of the time-slice of the CGM e.g. nodes=['X', 'Z', 'Y']
    target_node: str, optional
        If we are using a independent spatial topology then we need to specify the target node
    verbose : bool, optional
        To print the graph or not.

    Returns
    -------
    Union[MultiDiGraph, str]
        Returns the DOT format of the graph or a networkx object
    """

    assert start_time < stop_time
    assert topology in ["dependent", "independent"]
    assert nodes

    if topology == "independent":
        assert target_node is not None
        assert isinstance(target_node, str)

    ## Time-slice connections

    spatial_edges = []
    ranking = []
    # Check if target node is in the list of nodes, and if so remove it
    if topology == "independent":
        if target_node in nodes:
            nodes.remove(target_node)
        node_count = len(nodes)
        assert target_node not in nodes
        connections = node_count * "{}_{} -> {}_{}; "
        edge_pairs = list(sum([(item, target_node) for item in nodes], ()))
    else:
        node_count = len(nodes)
        connections = (node_count - 1) * "{}_{} -> {}_{}; "
        edge_pairs = [item for pair in list(zip(nodes, nodes[1:])) for item in pair]

    # print("edge_pairs:", edge_pairs)
    pair_count = len(edge_pairs)

    # cross-variable connections
    if topology == "independent":
        # X_0 --> Y_0; Z_0 --> Y_0
        all_nodes = nodes + [target_node]
        for t in range(start_time, stop_time ):
            space_idx = pair_count * [t]
            iters = [iter(edge_pairs), iter(space_idx)]
            inserts = list(chain(map(next, cycle(iters)), *iters))
            spatial_edges.append(connections.format(*inserts))
            ranking.append(
                "{{ rank=same; {} }} ".format(
                    " ".join([item + "_{}".format(t) for item in all_nodes])
                )
            )
    elif topology == "dependent":
        # X_0 --> Z_0; Z_0 --> Y_0
        for t in range(start_time, stop_time):
            space_idx = pair_count * [t]
            iters = [iter(edge_pairs), iter(space_idx)]
            inserts = list(chain(map(next, cycle(iters)), *iters))
            spatial_edges.append(connections.format(*inserts))
            ranking.append(
                "{{ rank=same; {} }} ".format(
                    " ".join([item + "_{}".format(t) for item in nodes])
                )
            )
    else:
        raise ValueError("Not a valid spatial topology.")

    ranking = "".join(ranking)
    spatial_edges = "".join(spatial_edges)

    ## Temporal connections / cross-time connections

    temporal_edges = []
    if topology == "independent":
        node_count += 1
        nodes += [target_node]

    connections = node_count * "{}_{} -> {}_{}; "
    
    # X_0 --> X_1; X_1 --> X_2; X_2 --> X_3
    for t in range(stop_time-1):
        edge_pairs = repeat(nodes, 2).tolist()
        temporal_idx = node_count * [t, t + 1]
        iters = [iter(edge_pairs), iter(temporal_idx)]
        inserts = list(chain(map(next, cycle(iters)), *iters))
        temporal_edges.append(connections.format(*inserts))

    temporal_edges = "".join(temporal_edges)

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


# if __name__ == "__main__":
#     G = make_graphical_model(
#         start_time=0,
#         stop_time=3,
#         topology="independent",
#         nodes=["X", "Z", "W", "Y"],
#         target_node="Y",
#         verbose=True,
#         filename="examples/indpt_graph",
#     )

    
    
#     G = make_graphical_model(
#         start_time=0,
#         stop_time=3,
#         topology="dependent",
#         nodes=["X", "Z", "W", "Y"],
#         target_node="Y",
#         verbose=True,
#         filename="examples/dpt_graph",
#     )
