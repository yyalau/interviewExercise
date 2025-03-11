from .base import PriorBase
import numpy as np
from data_struct import hDict
from models import GPRegression, KernelDensity
from tensorflow_probability import distributions as tfd


class PriorEmit(PriorBase):
    def __init__(self, G, dtype = "float32"):
        super().__init__(G, dtype)

    def fork_ops(self, pa_node, pa_value, ch_node, ch_value, i, funcs):
        """
        Fork node operations.
        """
        if pa_node.t == ch_node.t:
            funcs.add(key=(k := (pa_node.name, i, ch_node.name)))
            funcs[k][pa_node.t, 0] = GPRegression(
                kernel=self.K_func, feature_ndims=1, dtype = self.dtype
            ).fit(x=pa_value, y=ch_value)

        return funcs

    def source_ops(self, pa_node, pa_value, funcs):
        """
        Source node operations.
        """
        funcs.add(key=(k := (None, pa_node.name)))
        funcs[k][pa_node.t, 0] = KernelDensity(kernel=tfd.Normal, dtype = self.dtype).fit(
            pa_value[..., None]
        )

        return funcs

    def normal_ops(self, pa_node, pa_value, ch_node, ch_value, funcs):
        """
        Normal node operations.
        """
        assert pa_node.t == ch_node.t, "Time mismatch for emission nodes."
        funcs.add(key=(k := (pa_node.name,)))
        funcs[k][pa_node.t, 0] = GPRegression(kernel_fn=self.K_func, feature_ndims=1, dtype = self.dtype).fit(
            x=pa_value[..., None], y=ch_value
        )
        return funcs

    def collider_ops(self, pa_nodes, pa_values, ch_node, ch_value, funcs):
        """
        Collider node operations.
        """
        if pa_nodes[0].t == ch_node.t:
            funcs.add(key=(k := tuple(pa_node.name for pa_node in pa_nodes)))
            funcs[k][pa_nodes[0].t, 0] = GPRegression(
                kernel_fn=self.K_func, feature_ndims=len(pa_nodes), dtype = self.dtype
            ).fit(x=pa_values[..., None], y=ch_value)

        return funcs

    # override
    def source_node(self, A, data):

        funcs = hDict(
            variables=[],
            nT=self.nT,
            nTrials=1,
        )

        # nothing --> X
        pa_nodes = self.nodes[np.where(A.sum(axis=0) == 0)[0]]
        for pa_node in pa_nodes:
            pa_value = data[pa_node.name][pa_node.t, :]
            funcs.update(self.source_ops(pa_node, pa_value, funcs))

        return A, funcs

    def get_M(self):
        """
        Returns adjacency matrix but only for the emission terms.
        """
        M = super().get_M()
        A = np.zeros_like(M)
        for k in range(self.nT):
            A[
                k * self.nVar : (k + 1) * self.nVar, k * self.nVar : (k + 1) * self.nVar
            ] = M[0 : self.nVar, 0 : self.nVar]

        return A
