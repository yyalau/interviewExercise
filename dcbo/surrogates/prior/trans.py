from ..base import PriorBase
from networkx.linalg.graphmatrix import adjacency_matrix
import numpy as np
from data_struct import hDict
from models import GPRegression


class PriorTrans(PriorBase):
    def __init__(self, G):
        super().__init__(G)

    def fork_ops(self, pa_node, pa_value, ch_node, ch_value, i, funcs):
        """
        Fork node operations.
        """

        if pa_node.t != ch_node.t:
            funcs.add(key=(k := (pa_node.name, i, ch_node.name)))
            funcs[k][ch_node.t, 0] = GPRegression(
                kernel_fn=self.K_func, feature_ndims=1
            ).fit(x=pa_value, y=ch_value)

        return funcs

    def source_ops(self, pa_node, pa_value, funcs):
        """
        Source node operations.
        """
        return {}

    def normal_ops(self, pa_node, pa_value, ch_node, ch_value, funcs):
        """
        Normal node operations.
        """

        assert pa_node.t != ch_node.t, "Time should be different for transition nodes."

        funcs.add(key=(k := (pa_node.name,)))
        funcs[k][pa_node.t, 0] = GPRegression(kernel_fn=self.K_func, feature_ndims=1).fit(
            x=pa_value[..., None], y=ch_value
        )

        return funcs

    def collider_ops(self, pa_nodes, pa_values, ch_node, ch_value, funcs):
        """
        Collider node operations.
        """

        if pa_nodes[0].t != ch_node.t:
            funcs.add(key=(k := tuple(pa_node.name for pa_node in pa_nodes)))
            funcs[k][pa_nodes[0].t, 0] = GPRegression(
                kernel_fn=self.K_func, feature_ndims=len(pa_nodes)
            ).fit(x=pa_values[..., None], y=ch_value)

        return funcs

    # override
    def source_node(self, A, data):

        funcs = hDict(
            variables=[],
            nT=self.nT,
            nTrials=1,
        )

        return A, funcs

    def get_M(self):
        """
        Returns adjacency matrix but only for the emission terms.
        """

        M = super().get_M()
        A = np.zeros_like(M)
        for t in range(self.nT - 1):
            A[
                t * self.nVar : (t + 1) * self.nVar,
                (t + 1) * self.nVar : (t + 2) * self.nVar,
            ] = M[
                t * self.nVar : (t + 1) * self.nVar,
                (t + 1) * self.nVar : (t + 2) * self.nVar,
            ]
        return A
