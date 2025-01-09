from typing import Callable
from surrogates import PriorEmit, PriorTrans
from data_struct import GraphObj, hDict, Node
import numpy as np
from utils.tools import tnode2var, tvar2node


class SEMHat:
    def __init__(self, G: GraphObj, gp_emit, gp_trans):
        self.G = G
        self.nVar = G.nVar
        self.nT = G.nT
        self.nodes = G.nodes
        self.vs = G.variables

        self.gp_emit = gp_emit
        self.gp_trans = gp_trans

    def filter_pa_t(self, node: str, t: int) -> tuple:
        """Function that returns the parents of a node at a given time-slice."""
        node_s = node.gstr if isinstance(node, Node) else node
        x = []
        for pa in self.G.dag.predecessors(node_s):
            pa_v, pa_t = pa.split("_")
            if int(pa_t) == t:
                x.append(Node(pa_v, pa_t))
        return tuple(x)

    def get_edgekeys(self, node: Node, t: int) -> tuple:
        """Function that returns the parents of a node at a given time-slice.

        Parameters
        ----------
        node : Node
            The child node of interest (target node)
        t : int
            The index of interest

        Returns
        -------
        tuple
            A tuple which only contains nodes with index t
        """

        def __get_edgekeys(pa_nodes, ch_node, edge_keys, t):
            if len(pa_nodes) == 0:
                return pa_nodes

            if pa_nodes in edge_keys:
                return pa_nodes

            if rkey := tuple(reversed(pa_nodes)) in edge_keys:
                return rkey

            for key in edge_keys:
                p, _, c = key
                if p == pa_nodes[0] and c == ch_node:
                    return key

            raise ValueError(
                f"Edge not found between {pa_nodes} and {ch_node} at time {t}"
            )

        if t < 0:
            return ()

        node_s = node.gstr if isinstance(node, Node) else node
        pa_nodes = self.filter_pa_t(node_s, t)

        if node.t - 1 == t:
            keys = [
                tvar2node(var, t)
                for var, gp in self.gp_trans.f.items()
                if gp[t, 0] is not None
            ]
            return __get_edgekeys(pa_nodes, node_s, keys, t)

        keys = [
            tvar2node(var, t)
            for var, gp in self.gp_emit.f.items()
            if gp[t, 0] is not None
        ]
        return __get_edgekeys(pa_nodes, node_s, keys, t)

    def update_prior(self, data):
        # if new data is available, can update the prior
        self.gp_emit.fit(data)
        self.gp_trans.fit(data)

    def select_sample(self, samples, edge_key, t, n_samples):

        l_key = len(edge_key)

        # if edge_key is a fork
        if l_key == 3 and edge_key[1] == t:
            pa_node, _, _ = edge_key
            assert pa_node.t == t, "Time mismatch for prior/node and samples."
            return samples[pa_node.name][pa_node.t, :].reshape(n_samples, -1)

        # otherwise
        samp = []
        for pa_node in edge_key:
            assert pa_node.t == t, "Time mismatch for prior/nodes and samples."
            samp += [samples[pa_node.name][pa_node.t, :].reshape(n_samples, -1)]

        return np.hstack(samp)

    def get_kernel(
        self,
    ) -> Callable:
        #  Assigns the KDE for the marginal
        return lambda t, margin_id, n_samples: self.gp_emit.f[margin_id][t, 0].sample(
            n_samples
        )

    def get_gp_emit(self, moment: int) -> Callable:
        #  Within time-slice emission only
        
        def __get_gp_emit(t, _, emit_keys, sample, n_samples):
            samples = self.select_sample(sample, emit_keys, t, n_samples)[...,None]
            return self.gp_emit.f[tnode2var(emit_keys)][t, 0].predict(samples).numpy()[moment]
        
        return __get_gp_emit

    def get_gp_trans(self, moment: int) -> Callable:
        #  Only transition between time-slices (only valid for first-order Markov assumption)
        
        def __get_gp_trans(t, trans_keys, _, sample, n_samples):
            samples= self.select_sample(sample, trans_keys, t - 1, n_samples)[..., None]
            return self.gp_trans.f[tnode2var(trans_keys)][t - 1, 0].predict(samples).numpy()[moment]

        return __get_gp_trans

    def get_gp_both(self, moment: int) -> Callable:
        #  Transition plus within-slice emission(s)
        
        def __get_gp_both(t, trans_keys, emit_keys, sample, n_samples ):
            trans = self.get_gp_trans(moment)(t,trans_keys, emit_keys, sample, n_samples) 
            emit = self.get_gp_emit(moment)(t,trans_keys, emit_keys, sample, n_samples) 
            return trans + emit
        
        return __get_gp_both

    def static(self, moment: int):
        assert moment in [0, 1], moment

        f = hDict(
            variables=self.vs,
            nT=1,
            nTrials=1,
        )

        for node in self.nodes:
            if node.t != 0:
                continue
            f[node.name][node.t, 0] = (
                self.get_kernel()
                if self.G.dag.in_degree[node.gstr] == 0
                else self.get_gp_emit(moment)
            )
        return f

    def dynamic(self, moment: int):
        assert moment in [0, 1], moment

        f = hDict(
            variables=self.vs,
            nT=1,
            nTrials=1,
        )

        for node in self.nodes:
            if node.t != 1:
                continue

            # Single source node
            if self.G.dag.in_degree[node.gstr] == 0:
                f[node.name][0, 0] = self.get_kernel()
                continue

            # Depends only on incoming transition edge(s)
            if all(
                int(pre_n.split("_")[1]) + 1 == node.t
                for pre_n in self.G.dag.predecessors(node.gstr)
            ):
                f[node.name][0, 0] = self.get_gp_trans(moment)
                continue

            # Depends only on incoming emission edge(s) from this time-slice
            if all(
                int(pre_n.split("_")[1]) == node.t
                for pre_n in self.G.dag.predecessors(node.gstr)
            ):
                f[node.name][0, 0] = self.get_gp_emit(moment)
                continue

            # Depends incoming emission AND transition edges
            f[node.name][0, 0] = self.get_gp_both(moment)

        return f

