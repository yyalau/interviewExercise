import pytest
import networkx as nx
from src.data_struct.graph_obj import GraphObj
from src.data_struct.node import Var, Node

def create_test_graph():
    graph = nx.DiGraph()
    graph.add_edges_from([
        ("A_0", "A_1"), ("A_1", "A_2"),
        ("B_0", "B_1"), ("B_1", "B_2"),
        ("C_0", "C_1"), ("C_1", "C_2"),
        ("A_0", "B_1"), ("B_1", "C_2")
    ])
    return graph

def test_graph_obj_initialization():
    graph = create_test_graph()
    target_var = Var(name="A")
    nT = 3
    graph_obj = GraphObj(graph, nT, target_var)
    
    assert graph_obj.nT == nT
    assert graph_obj.target_variable == target_var
    assert graph_obj.nVar == 3
    assert len(graph_obj.nodes) == 9
    assert len(graph_obj.variables) == 3
    


def test_invalid_nT():
    graph = create_test_graph()
    target_var = Var(name="A")
    with pytest.raises(AssertionError):
        GraphObj(graph, 0, target_var)

@pytest.mark.parametrize("target_var", ["A", Var("D"), Node("A", 0)])
def test_invalid_target_var(target_var):
    graph = create_test_graph()
    with pytest.raises(AssertionError):
        GraphObj(graph, 3, target_var)


@pytest.mark.parametrize("edges", [
        ("A_2", "A_0"),
        ("C_2", "A_0"),
        ("C_2", "A"),
        ("A", "A_0"),
        ("B_2", "B_3"),
        ("C_2", "D_2"),
])
def test_invalid_graph(edges):
    graph = create_test_graph()
    graph.add_edge(*edges)  # Adding a cycle
    target_var = Var(name="A")
    with pytest.raises(AssertionError):
        GraphObj(graph, 3, target_var)