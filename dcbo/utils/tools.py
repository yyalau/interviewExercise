from itertools import chain, combinations
from data_struct import Node, Var

def powerset(iterable):
    # this returns e.g. powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

def tnode2var(tnode):
    return tuple([node.name if node is not None else None for node in tnode])

def tvar2node(tvar,t):
    return tuple([Node(var.name, t) if var is not None else None for var in tvar ])