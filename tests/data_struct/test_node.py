from data_struct.node import Var, Node
import pytest

def test_var_initialization():
    var = Var("test")
    assert var.name == "test"
    assert str(var) == "test"
    

def test_var_eq():
    var1 = Var("test")
    var2 = Var("test")
    assert var1 == var2

    var1 = Var("test0")
    var2 = Var("test")
    assert var1 != var2

def test_var_hash():
    var = Var("test")
    assert hash(var) == hash("test")

@pytest.mark.parametrize("name", [123, "a_1", "_", ""])
def test_var_invalid_name(name):
    with pytest.raises(AssertionError):
        Var(name)
        
def test_node_initialization():
    node = Node("test", 1)
    assert node.name == Var("test")
    assert node.t == 1
    assert node.gstr == "test_1"

def test_node_str():
    node = Node("test", 1)
    assert str(node) == "test_1"


def test_node_eq():
    node1 = Node("test", 1)
    node2 = Node("test", 1)
    assert node1 == node2
    assert node1 == "test_1"

def test_node_neq():
    node1 = Node("test", 1)
    node2 = Node("test", 2)
    assert node1 != node2

    node1 = Node("test0", 1)
    node2 = Node("test", 1)
    assert node1 != node2


def test_node_hash():
    node = Node("test", 1)
    assert hash(node) == hash((Var("test"), 1))

@pytest.mark.parametrize("name", [123, "a_1", "_",""])
def test_node_invalid_name(name):
    with pytest.raises(AssertionError):
        Node(name, 1)

@pytest.mark.parametrize("t", ["1", 1.0, -1])
def test_node_invalid_t(t):
    with pytest.raises(AssertionError):
        Node("test", t)
