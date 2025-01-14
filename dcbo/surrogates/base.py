from data_struct import hDict, Node
import numpy as np
from networkx.linalg.graphmatrix import adjacency_matrix
import tensorflow_probability as tfp

RBF = tfp.math.psd_kernels.ExponentiatedQuadratic

class PriorBase:
    def __init__(self, G):
        self.G = G
        self.nT = G.nT        
        self.nVar = G.nVar
        self.nodes = G.nodes
        
        self.f = hDict(
            variables=[],
            nT=self.nT,
            nTrials=1,
        )
        
        self.K_func = RBF        
    
    def fit(self, data):
        """
        Returns the emission functions.
        """
        A = self.get_M(); B = A.copy()
        A, fork_func = self.fork_node(A, data)
        A, source_func = self.source_node(A, data)
        A, normal_func = self.normal_node(A, data)
        
        assert A.sum() == 0, "There are still edges in the adjacency matrix."
        _, collider_func = self.collider_node(B, data)
        
        self.f.update(fork_func, source_func, normal_func, collider_func)
        return self.f

    def __empty_hDict(self,):
        return hDict(
            variables=[],
            nT=self.nT,
            nTrials=1,
        )
    
    # def fork_ops(self,):
    #     return self.__empty_hDict()
    
    # def normal_ops(self,):
    #     return self.__empty_hDict()

    # def source_ops(self,):
    #     return self.__empty_hDict()
    
    # def collider_ops(self,):
    #     return self.__empty_hDict()

    def fork_node(self, A, data):
        
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
                funcs.update(self.fork_ops(pa_node, pa_value, ch_node, ch_value, i, funcs))
                
                A[pa_i, ch_i] -= 1
        
        return A, funcs


    def source_node(self, A, data):
        raise NotImplementedError("Source node not implemented.")
    
    
    def normal_node(self, A, data):
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


    def collider_node(self, A, data):
        funcs = self.__empty_hDict()

        # X --> Y <-- Z
        ch_idx = np.where(A.sum(axis=0) > 1)[0]  # idx of Y
        ch_nodes = self.nodes[ch_idx]
        
        for ch_i, ch_node in zip(ch_idx, ch_nodes):
            
            pa_idx = np.where(A[:, ch_i] == 1)
            pa_nodes = self.nodes[pa_idx]
            
            pa_value = np.hstack([data[pa_node.name][pa_node.t, :, None] for pa_node in pa_nodes])
            ch_value = data[ch_node.name][ch_node.t, :]
            
            funcs.update(self.collider_ops(pa_nodes, pa_value, ch_node, ch_value,funcs))

        
        return A, funcs
    
    def get_M(self):
        return adjacency_matrix(self.G.dag).todense()