from itertools import chain, combinations
from data_struct import Node, Var
import tensorflow as tf
import numpy as np


def powerset(iterable):
    # this returns e.g. powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def tnode2var(tnode):
    return tuple([node.name if isinstance(node, Node)  else node for node in tnode])


def tvar2node(var_keys, t1, t2):
    result = []
    count = 0 
    for i, element in enumerate(var_keys):
        if not isinstance(element, Var):
            result.append(element)
            count +=1 
            continue
        result.append(Node(element, t1 if count ==0 else t2) )
    return tuple(result)

def get_probH0(x, y, m0, m1, dtype="float32"):
    y0 = m0.prob(y)
    y1 = m1.prob(y, x=x)
    return tf.reduce_mean(tf.cast(y0 > y1, dtype)).numpy()


def update_posteriorH0(D_Int, m0, m1, prior_H0):

    prob_H0 = get_probH0(D_Int.dataX, D_Int.dataY, m0, m1, dtype=D_Int.dtype)
    prob_H1 = 1 - prob_H0

    base = prob_H0 * prior_H0 + prob_H1 * (1 - prior_H0)
    return prob_H0 * prior_H0 / base


def eager_replace(arr, new_val, idx, axis=0, dtype="float32"):

    new_arr = (
        tf.convert_to_tensor(new_val.astype(dtype))
        if isinstance(new_val, np.ndarray)
        else tf.cast(new_val, dtype)
    )

    return tf.stack(
        [(arr[tempt] if tempt != idx else new_arr) for tempt in range(arr.shape[axis])],
        axis=axis,
    )
