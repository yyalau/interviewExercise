from data_struct import hDict, GraphObj
import numpy as np
from networkx.linalg.graphmatrix import adjacency_matrix
import tensorflow_probability as tfp
from typing import Tuple

RBF = tfp.math.psd_kernels.ExponentiatedQuadratic


class PriorBase:
    def __init__(self, G: GraphObj, dtype: str = "float32"):
        """
        The base class for the prior emission / transition functions. Generates a hDict object to store the functions, according to the graph structure.

        Parameters:
        -----------
        G: GraphObj
            The graph object.
        dtype: str
            The data type of the emission functions.
        """

        assert isinstance(G, GraphObj), f"Expected GraphObj, got {type(G)}"
        assert isinstance(
            dtype, str
        ), f"Expected dtype to be of type str, got {type(dtype)}"

        self.G = G
        self.nT = G.nT
        self.nVar = G.nVar
        self.nodes = G.nodes
        self.dtype = dtype
        self.variables = G.variables

        self.f = hDict(
            variables=[],
            nT=self.nT,
            nTrials=1,
        )

        self.K_func = RBF

    def fit(self, data: hDict) -> hDict:
        """
        Returns the prior functions according to the graph structure.

        Parameters:
        -----------
        data: hDict
            The data dictionary. The keys must match the variable names in the graph.
            The values must be arrays or tensors of shape (nT, ...).
        
        Returns:
        --------
        hDict
            The prior functions in a hDict object.
        ```
        """
        assert isinstance(data, hDict), f"Expected hDict, got {type(data)}"
        for var in self.variables:
            assert var in data, f"Expected variable {var} in data."
            assert (
                data[var].shape[0] == self.nT
            ), f"Expected data to have shape ({self.nT}, ...), got {data[var].shape}"

        A = self.get_M()
        B = A.copy()

        A, fork_func = self.fork_node(A, data)
        A, source_func = self.source_node(A, data)
        A, normal_func = self.normal_node(A, data)

        assert A.sum() == 0, "There are still edges in the adjacency matrix."
        _, collider_func = self.collider_node(B, data)

        for x in [fork_func, source_func, normal_func, collider_func]:
            self.f.update(x)
        return self.f

    def __empty_hDict(self) -> hDict:
        '''
        Returns an empty hDict object.
        '''
        
        return hDict(
            variables=[],
            nT=self.nT,
            nTrials=1,
        )

    def fork_node(self, A: np.ndarray, data: hDict) -> Tuple[np.ndarray, hDict]:
        '''
        Fork node operations. e.g. Y <-- X --> Z
        
        Parameters:
        -----------
        A: np.ndarray
            The adjacency matrix.
        data: hDict
            The data dictionary. The keys must match the variable names in the graph.
            The values must be arrays or tensors of shape (nT, ...).
        
        Returns:
        --------
        A: np.ndarray
            The updated adjacency matrix with the fork edges removed.        
        funcs: hDict
            The functions derived from the fork nodes.
        '''
        
        funcs = self.__empty_hDict()

        # Y <-- X --> Z
        pa_idx = np.where(A.sum(axis=1) > 1)[0]  # list of all fork parents X
        pa_nodes = self.nodes[pa_idx]

        for pa_i, pa_node in zip(pa_idx, pa_nodes):
            pa_value = data[pa_node.name][pa_node.t, :].reshape(-1, 1)

            ch_idx = np.where(A[pa_i, :] == 1)[0]  # idx of Y and Z
            ch_nodes = self.nodes[ch_idx]

            for i, (ch_i, ch_node) in enumerate(zip(ch_idx, ch_nodes)):

                ch_value = data[ch_node.name][ch_node.t, :].reshape(-1, 1)
                funcs.update(
                    self.fork_ops(pa_node, pa_value, ch_node, ch_value, i, funcs)
                )

                A[pa_i, ch_i] -= 1

        return A, funcs

    def source_node(self, A: np.ndarray, data: hDict) -> Tuple[np.ndarray, hDict]:
        '''
        Source node operations. e.g. X. It has to be implemented in the derived class.
        
        Parameters:
        -----------
        A: np.ndarray
            The adjacency matrix.
        data: hDict
            The data dictionary. The keys must match the variable names in the graph.
            The values must be arrays or tensors of shape (nT, ...).
        
        Returns:
        --------
        A: np.ndarray
            The updated adjacency matrix with the source nodes removed.        
        funcs: hDict
            The functions derived from the source nodes.        
        '''
        raise NotImplementedError("Source node not implemented.")

    def normal_node(self, A: np.ndarray, data: hDict) -> Tuple[np.ndarray, hDict]:
        '''
        Normal node operations. e.g. X --> Y
        
        Parameters:
        -----------
        A: np.ndarray
            The adjacency matrix.
        data: hDict
            The data dictionary. The keys must match the variable names in the graph.
            The values must be arrays or tensors of shape (nT, ...).
            
        Returns:
        --------
        A: np.ndarray
            The updated adjacency matrix with the normal edges removed.        
        funcs: hDict
            The functions derived from the normal nodes.
        '''
        funcs = self.__empty_hDict()
        # X --> Y
        for pa_idx, ch_idx in zip(*np.where(A == 1)):
            pa_node = self.nodes[pa_idx]
            ch_node = self.nodes[ch_idx]
            pa_value = data[pa_node.name][pa_node.t, :]
            ch_value = data[ch_node.name][ch_node.t, :]

            funcs.update(self.normal_ops(pa_node, pa_value, ch_node, ch_value, funcs))
            A[pa_idx, ch_idx] -= 1

        return A, funcs

    def collider_node(self, A: np.ndarray, data: hDict) -> Tuple[np.ndarray, hDict]:
        '''
        Collider node operations. e.g. X <-- Y --> Z
        
        Parameters:
        -----------
        A: np.ndarray
            The adjacency matrix.
        data: hDict
            The data dictionary. The keys must match the variable names in the graph.
            The values must be arrays or tensors of shape (nT, ...).
        
        Returns:
        --------
        A: np.ndarray
            The updated adjacency matrix with the collider edges removed.
        funcs: hDict
            The functions derived from the collider nodes.
        '''
        funcs = self.__empty_hDict()

        # X --> Y <-- Z
        ch_idx = np.where(A.sum(axis=0) > 1)[0]  # idx of Y
        ch_nodes = self.nodes[ch_idx]

        for ch_i, ch_node in zip(ch_idx, ch_nodes):

            pa_idx = np.where(A[:, ch_i] == 1)
            pa_nodes = self.nodes[pa_idx]

            pa_value = np.hstack(
                [data[pa_node.name][pa_node.t, :, None] for pa_node in pa_nodes]
            )
            ch_value = data[ch_node.name][ch_node.t, :]

            funcs.update(
                self.collider_ops(pa_nodes, pa_value, ch_node, ch_value, funcs)
            )
            A[pa_idx, ch_idx] -= 1

        return A, funcs

    def get_M(self) -> np.ndarray:
        '''
        Returns the adjacency matrix according to the graph structure.
        '''
        gstr = [node.gstr for node in self.nodes]
        return adjacency_matrix(self.G.dag, nodelist=gstr).todense()
