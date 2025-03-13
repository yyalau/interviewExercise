from .base import PriorBase
import numpy as np
from data_struct import hDict, Node, Var
from models import GPRegression, KernelDensity
from tensorflow_probability import distributions as tfd
from typing import List, Union, Tuple
import tensorflow as tf

class PriorEmit(PriorBase):
    def __init__(self, G, dtype: str = "float32"):
        '''
        The class for the prior emission functions. Inherits from PriorBase.         
        '''
        super().__init__(G, dtype)

    def fork_ops(self, pa_node: Node, pa_value: Union[np.ndarray, tf.Tensor], ch_node: Node, ch_value: Union[np.ndarray, tf.Tensor], i: int, funcs: hDict[Tuple[Var, int, Var], GPRegression]) -> hDict[Tuple[Var, int, Var], GPRegression]:
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
        
        if pa_node.t == ch_node.t:
            funcs.add(key=(k := (pa_node.name, i, ch_node.name)))
            funcs[k][pa_node.t, 0] = GPRegression(
                kernel=self.K_func, feature_ndims=1, dtype=self.dtype
            ).fit(x=pa_value, y=ch_value)

        return funcs

    def source_ops(self, pa_node: Node, pa_value: Union[np.ndarray, tf.Tensor], funcs: hDict[Tuple[None, Var], KernelDensity]) -> hDict[Tuple[None, Var], KernelDensity]:
        """
        Source node operations.
        
        Parameters:
        -----------
        pa_node: Node
            The parent node.
        pa_value: np.ndarray or tf.Tensor
            The parent node values.
        funcs: hDict[Tuple[None, Var], KernelDensity] or hDict[None]
            Dictionary of functions representing the graph structure. If the dictionary is not empty, the values are KernelDensity objects. The keys for source nodes are tuples of the form (None, var).
                    
        Returns:
        --------
        funcs: hDict[Tuple[None, Node], KernelDensity]
            The updated dictionary of functions representing the graph structure.
        """
        assert isinstance(pa_node, Node), "The parent node must be an instance of Node."
        assert isinstance(pa_value, (np.ndarray, tf.Tensor)), "The parent node value must be an array or tensor."
        assert isinstance(funcs, hDict), "The functions must be an instance of hDict."
        assert pa_value.ndim == 1, "The parent node value must be a 1D array."

        for key in funcs.keys():
            assert isinstance(key, tuple), "The keys must be tuples."
            assert len(key) == 2, "The keys must be of length 2."
            assert key[0] is None, "The first element of the key must be None."
            assert isinstance(key[1], Var), "The second element of the key must be a Node."

        funcs.add(key=(k := (None, pa_node.name)))
        funcs[k][pa_node.t, 0] = KernelDensity(kernel=tfd.Normal, dtype=self.dtype).fit(
            pa_value[..., None]
        )

        return funcs

    def normal_ops(self, pa_node: Node, pa_value: Union[np.ndarray, tf.Tensor], ch_node: Node, ch_value: Union[np.ndarray, tf.Tensor], funcs: hDict[Tuple[Var, Var], GPRegression]) -> hDict[Tuple[Var, Var], GPRegression]:
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
        assert pa_node.t == ch_node.t, "Time mismatch for emission nodes."
        funcs.add(key=(k := (pa_node.name, )))
        funcs[k][pa_node.t, 0] = GPRegression(kernel_fn=self.K_func, feature_ndims=1, dtype=self.dtype).fit(
            x=pa_value[..., None], y=ch_value
        )
        return funcs

    def collider_ops(self, pa_nodes: Union[List[Node], np.ndarray], pa_values: Union[np.ndarray, tf.Tensor], ch_node: Node, ch_value: Union[np.ndarray, tf.Tensor], funcs: hDict[Tuple[Var, ...], GPRegression]) -> hDict[Tuple[Var, ...], GPRegression]:
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
        if pa_nodes[0].t == ch_node.t:
            funcs.add(key=(k := tuple(pa_node.name for pa_node in pa_nodes)))
            funcs[k][pa_nodes[0].t, 0] = GPRegression(
                kernel_fn=self.K_func, feature_ndims=len(pa_nodes), dtype=self.dtype
            ).fit(x=pa_values[..., None], y=ch_value)

        return funcs

    # override
    def source_node(self, A: np.ndarray, data: hDict[Var, np.ndarray]) -> Tuple[np.ndarray, hDict]:
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

        pa_nodes = self.nodes[np.where(A.sum(axis=0) == 0)[0]]
        for pa_node in pa_nodes:
            pa_value = data[pa_node.name][pa_node.t, :]
            funcs.update(self.source_ops(pa_node, pa_value, funcs))

        return A, funcs

    def get_M(self) -> np.ndarray:
        """
        Returns adjacency matrix but only for the emission terms.
        
        Returns:
        --------
        A: np.ndarray
            The adjacency matrix for the emission terms.
        """
        M = super().get_M()
        A = np.zeros_like(M)
        for k in range(self.nT):
            A[
                k * self.nVar : (k + 1) * self.nVar, k * self.nVar : (k + 1) * self.nVar
            ] = M[0 : self.nVar, 0 : self.nVar]

        return A
