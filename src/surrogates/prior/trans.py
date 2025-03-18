from .base import PriorBase
from networkx.linalg.graphmatrix import adjacency_matrix
import numpy as np
from data_struct import hDict, Node, Var
from models import GPRegression
from typing import List, Union, Tuple
import tensorflow as tf


class PriorTrans(PriorBase):
    def __init__(self, G, dtype: str = "float32"):
        """
        The class for generating prior transition functions. Inherits from PriorBase.
        """
        super().__init__(G, dtype)

    def fork_ops(
        self,
        pa_node: Node,
        pa_value: Union[np.ndarray, tf.Tensor],
        ch_node: Node,
        ch_value: Union[np.ndarray, tf.Tensor],
        i: int,
        funcs: hDict[Tuple[Var, int, Var], GPRegression],
    ) -> hDict[Tuple[Var, int, Var], GPRegression]:
        """
        Fork node operations.

        Parameters:
        -----------
        pa_node: Node
            The parent node.
        pa_value: np.ndarray or tf.Tensor
            The parent node values.
        ch_node: Node
            The child node.
        ch_value: np.ndarray or tf.Tensor
            The child node values.
        i: int
            The i-th connection between the parent and child nodes.
        funcs: hDict[Tuple[Var, int, Var], GPRegression] or hDict[None]
            Dictionary of functions representing the graph structure. If the dictionary is not empty, the values are GPRegression objects. The keys for fork nodes are tuples of the form (parent_var, i, child_var).

        Returns:
        --------
        funcs: hDict[Tuple[Var, int, Var], GPRegression]
            The updated dictionary of functions representing the graph structure. The keys for fork nodes are tuples of the form (parent_var, i, child_var). The values are GPRegression objects.
        """

        if pa_node.t != ch_node.t:
            funcs.add(key=(k := (pa_node.name, i, ch_node.name)))
            funcs[k][pa_node.t, 0] = GPRegression(
                kernel_fn=self.K_func, feature_ndims=1, dtype=self.dtype
            ).fit(x=pa_value[..., None], y=ch_value)

        return funcs

    def source_ops(
        self,
        pa_node: Node,
        pa_value: Union[np.ndarray, tf.Tensor],
        funcs: hDict[Tuple[None, Var], GPRegression],
    ) -> hDict[Tuple[None, Var], GPRegression]:
        """
        Source node operations. No implementation of source node operations in transmission terms, since they do not have any parent nodes and time differences are not considered.

        Parameters:
        -----------
        pa_node: Node
            The parent node.
        pa_value: np.ndarray or tf.Tensor
            The parent node values.
        funcs: hDict[Tuple[None, Var], GPRegression] or hDict[None]
            Dictionary of functions representing the graph structure. If the dictionary is not empty, the values are GPRegression objects. The keys for source nodes are tuples of the form (None, var).

        Returns:
        --------
        funcs: hDict[Tuple[None, Var], GPRegression]
            The updated dictionary of functions representing the graph structure.
        """
        return {}

    def normal_ops(
        self,
        pa_node: Node,
        pa_value: Union[np.ndarray, tf.Tensor],
        ch_node: Node,
        ch_value: Union[np.ndarray, tf.Tensor],
        funcs: hDict[Tuple[Var, Var], GPRegression],
    ) -> hDict[Tuple[Var, Var], GPRegression]:
        """
        Normal node operations.

        Parameters:
        -----------
        pa_node: Node
            The parent node.
        pa_value: np.ndarray or tf.Tensor
            The parent node values.
        ch_node: Node
            The child node.
        ch_value: np.ndarray or tf.Tensor
            The child node values.
        funcs: hDict[Tuple[Var, Var], GPRegression] or hDict[None]
            Dictionary of functions representing the graph structure. If the dictionary is not empty, the values are GPRegression objects. The keys for normal nodes are tuples of the form (parent_var, child_var).

        Returns:
        --------
        funcs: hDict[Tuple[Var, Var], GPRegression]
            The updated dictionary of functions representing the graph structure. The keys for normal nodes are tuples of the form (parent_var, child_var). The values are GPRegression objects.
        """

        if pa_node.t != ch_node.t:

            funcs.add(key=(k := (pa_node.name,)))
            funcs[k][pa_node.t, 0] = GPRegression(
                kernel_fn=self.K_func, feature_ndims=1, dtype=self.dtype
            ).fit(x=pa_value[..., None], y=ch_value)

        return funcs

    def collider_ops(
        self,
        pa_nodes: Union[List[Node], np.ndarray],
        pa_values: Union[np.ndarray, tf.Tensor],
        ch_node: Node,
        ch_value: Union[np.ndarray, tf.Tensor],
        funcs: hDict[Tuple[Var, ...], GPRegression],
    ) -> hDict[Tuple[Var, ...], GPRegression]:
        """
        Collider node operations.

        Parameters:
        -----------
        pa_nodes: List[Node] or np.ndarray[Node]
            The parent nodes.
        pa_values: np.ndarray or tf.Tensor
            The parent node values.
        ch_node: Node
            The child node.
        ch_value: np.ndarray or tf.Tensor
            The child node values.
        funcs: hDict[Tuple[Var, ...], GPRegression] or hDict[None]
            Dictionary of functions representing the graph structure. If the dictionary is not empty, the values are GPRegression objects. The keys for collider nodes are tuples of the form (parent_var1, parent_var2, ..., child_var).

        Returns:
        --------
        funcs: hDict[Tuple[Var, ...], GPRegression]
            The updated dictionary of functions representing the graph structure. The keys for collider nodes are tuples of the form (parent_var1, parent_var2, ..., child_var). The values are GPRegression objects.
        """
        for pa_node in pa_nodes:
            assert (
                pa_node.t != ch_node.t
            ), "Time should be different for transition nodes."

        pa_nodes = sorted(pa_nodes, key=lambda x: x.gstr)
        funcs.add(
            key=(
                k := tuple(
                    pa_node.name for pa_node in pa_nodes
                )
            )
        )
        funcs[k][pa_nodes[0].t, 0] = GPRegression(
            kernel_fn=self.K_func,
            feature_ndims=2 if len(pa_nodes) != 1 else 1,
            dtype=self.dtype,
        ).fit(x=pa_values[..., None], y=ch_value)

        return funcs

    # override
    def source_node(self, A, data):
        """
        Source node operations.

        Parameters:
        -----------
        A: np.ndarray
            The adjacency matrix.
        data: hDict[Var, np.ndarray]
            The data dictionary. The keys must match the variable names in the graph.
            The values must be arrays or tensors of shape (nT, ...).

        Returns:
        --------
        A: np.ndarray
            The updated adjacency matrix with the source edges removed.
        funcs: hDict
            The functions derived from the source nodes.
        """
        funcs = hDict(
            variables=[],
            nT=self.nT,
            nTrials=1,
        )

        return A, funcs

    def get_M(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns adjacency matrix but only for the transition terms.

        Returns:
        --------
        M: np.ndarray
            The adjacency matrix for the graph structure.
        A: np.ndarray
            The adjacency matrix for the transition terms.
        """
        M = self.get_adj()
        A = M.copy()
        for t in range(self.nT):
            A[
                t * self.nVar : (t + 1) * self.nVar, t * self.nVar : (t + 1) * self.nVar
            ] = 0

        return A
