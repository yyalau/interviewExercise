from typing import Callable, Tuple, List, Sequence, Union
from data_struct import GraphObj, hDict, Node
import numpy as np
from utils.tools import tnode2var, tvar2node
import tensorflow as tf
from .prior import PriorEmit, PriorTrans


class SEMHat:
    def __init__(self, G: GraphObj, dataY: hDict, dtype: str = "float32"):
        '''
        Initializes the SEMHat object.
        
        Parameters:
        -----------
        G: GraphObj
            The GraphObj object which contains the graph structure.
        dataY: hDict
            The data dictionary that contains the data of the variables.
        dtype: str
            The data type of the surrogate. Either "float32" or "float64".
        '''
        assert isinstance(G, GraphObj), f"Expected GraphObj, got {type(G)}"
        assert isinstance(dataY, hDict), f"Expected hDict, got {type(dataY)}"
        assert set(dataY.keys()) == set(G.variables), "Data keys must match the graph variables."
        for key, val in dataY.items():
            assert isinstance(val, np.ndarray), f"Expected np.ndarray, got {type(val)}"
            assert val.shape[0] == G.nT, "Data shape must match the number of time-slices."
        assert dtype in ["float32", "float64"], f"Expected 'float32' or 'float64', got {dtype}"
        
        self.G = G
        self.nVar = G.nVar
        self.nT = G.nT
        self.nodes = G.nodes
        self.vs = G.variables
        
        self.gp_emit = PriorEmit(G, dtype=dtype).fit(dataY)
        self.gp_trans = PriorTrans(G, dtype=dtype).fit(dataY)
        self.dtype = dtype

    def filter_pa_t(self, node: Union[str, Node], t: int) -> Tuple[Node, ...]:
        '''
        Returns the parents of a node at a given time-slice.
        
        Parameters:
        -----------
        node: str or Node
            The child node of interest
        t: int
            The index of interest        
        '''
        assert isinstance(t, int), "Time index must be an integer."
        assert t >= 0, "Time index must be greater than or equal to 0."
        assert t < self.nT, "Time index must be less than nT."
        assert isinstance(node, (str, Node)), "Node must be a string or Node object."
        
        node_s = node.gstr if isinstance(node, Node) else node
        x = []
        for pa in self.G.dag.predecessors(node_s):
            pa_v, pa_t = pa.split("_")
            if int(pa_t) == t:
                x.append(Node(pa_v, pa_t))
        return tuple(x)

    def get_edgekeys(self, node: Node, t: int) -> Tuple:
        """
        Function that returns the parents of a node at a given time-slice.

        Parameters
        ----------
        node : Node
            The child node of interest 
        t : int
            The index of interest

        Returns
        -------
        tuple
            A tuple which only contains nodes with index t
        """
        
        assert isinstance(t, int), "Time index must be an integer."
        if t < 0:
            return ()
                
        assert t < self.nT, "Time index must be less than nT."
        assert isinstance(node, Node), "Node must be a Node object."

        def __get_edgekeys(pa_nodes: Tuple, ch_node: Node, edge_keys: List, t: int) -> Tuple:
            if len(pa_nodes) == 0:
                return pa_nodes

            if pa_nodes in edge_keys:
                return pa_nodes

            if rkey := tuple(reversed(pa_nodes)) in edge_keys:
                return rkey

            for key in edge_keys:
                if len(key) != 3:
                    continue
                p, _, c = key
                if p == pa_nodes[0] and c == ch_node:
                    return key
                
            raise ValueError(
                f"Edge not found between {pa_nodes} and {ch_node} at time {t}"
            )


        node_s = node.gstr if isinstance(node, Node) else node
        pa_nodes = self.filter_pa_t(node_s, t)

        if node.t - 1 == t:
            keys = [
                tvar2node(var, t, t+1)
                for var, gp in self.gp_trans.f.items()
                if gp[t, 0] is not None
            ]
            return __get_edgekeys(pa_nodes, node_s, keys, t)

        keys = [
            tvar2node(var, t, t)
            for var, gp in self.gp_emit.f.items()
            if gp[t, 0] is not None
        ]
        return __get_edgekeys(pa_nodes, node_s, keys, t)

    def update_prior(self, data: hDict) -> None:
        '''
        Updates the prior of the SEMHat object.
        
        Parameters:
        -----------
        data: hDict
            The data dictionary that contains the data of the variables.
        '''
        assert isinstance(data, hDict), f"Expected hDict, got {type(data)}"
        self.gp_emit.fit(data)
        self.gp_trans.fit(data)

    def select_sample(self, samples: hDict, edge_key: Sequence, t: int, n_samples: int) -> tf.Tensor:
        """
        Function that selects the samples from the dictionary.
        
        Parameters
        ----------
        samples : hDict
            The dictionary that contains the samples.
        edge_key : Sequence
            The edge key of the node.
        t : int
            The time index.
        n_samples : int
            The number of samples. 
        """
        assert isinstance(samples, hDict), f"Expected hDict, got {type(samples)}"
        assert isinstance(edge_key, Sequence), f"Expected Sequence, got {type(edge_key)}"
        assert isinstance(t, int), f"Expected int, got {type(t)}"
        assert isinstance(n_samples, int), f"Expected int, got {type(n_samples)}"
        assert t >= 0, "Time index must be greater than or equal to 0."
        assert t < self.nT, "Time index must be less than nT."
        assert n_samples > 0, "Number of samples must be greater than 0."
        
        l_key = len(edge_key)

        # if edge_key is a fork
        if l_key == 3 and edge_key[1] == t:
            pa_node, _, _ = edge_key
            assert pa_node.t == t, "Time mismatch for prior/node and samples."
            return tf.cast(
                tf.reshape(samples[pa_node.name][pa_node.t], (n_samples, -1)),
                self.dtype,
            )

        # otherwise
        samp = []
        for pa_node in edge_key:
            assert pa_node.t == t, "Time mismatch for prior/nodes and samples."
            samp += [
                tf.cast(
                    tf.reshape(samples[pa_node.name][pa_node.t], (n_samples, -1)),
                    self.dtype,
                )
            ]
        return tf.reshape(tf.stack(samp, axis=1), (n_samples, -1))

    def get_kernel(self) -> Callable[[int, int, int], tf.Tensor]:
        '''
        Assigns the kernel density estimator for the marginal.
        
        Returns:
        --------
        Callable
            The kernel density estimator function, which takes in (the time-slice, the margin id, and the number of samples) as parameters and outputs the prediction from KDEs.
        '''
        return lambda t, margin_id, n_samples: tf.reshape(self.gp_emit.f[margin_id][t, 0].sample(
            n_samples
        ), (-1,))

    def get_gp_emit(self, moment: int) -> Callable:
        '''
        Returns the Gaussian Process functions for the emission edges.
        
        Parameters:
        -----------
        moment: int
            The moment of interest. Either 0 (for mean) or 1 (for variance).
        
        Returns:
        --------
        Callable
            The Gaussian Process function for the emission edges, which takes in (the time-slice, the emission keys, the transition keys, the values for prediction, and the number of samples) as parameters and outputs the prediction from GP / KDEs.
        '''
        assert moment in [0, 1], "Moment must be either 0 or 1."
        def __get_gp_emit(t: int, _: Sequence, emit_keys: Sequence, sample: hDict, n_samples: int) -> tf.Tensor:
            samples = self.select_sample(sample, emit_keys, t, n_samples)
            return tf.reshape(self.gp_emit.f[tnode2var(emit_keys)][t, 0].predict(samples)[moment], (-1,))

        return __get_gp_emit

    def get_gp_trans(self, moment: int) -> Callable:
        '''
        Returns the Gaussian Process functions for the transition edges.
        
        Parameters:
        -----------
        moment: int
            The moment of interest. Either 0 (for mean) or 1 (for variance).

        Returns:
        --------
        Callable
            The Gaussian Process function for the transition edges, which takes in (the time-slice, the transition keys, the emission keys, the values for prediction, and the number of samples) as parameters and outputs the prediction from GP / KDEs.
        '''
        
        assert moment in [0, 1], "Moment must be either 0 or 1."
        def __get_gp_trans(t: int, trans_keys: Sequence, _: Sequence, sample: hDict, n_samples: int) -> tf.Tensor:
            samples = self.select_sample(sample, trans_keys, t - 1, n_samples)
            return tf.reshape(self.gp_trans.f[tnode2var(trans_keys)][t - 1, 0].predict(samples)[
                moment
            ], (-1,))

        return __get_gp_trans

    def get_gp_both(self, moment: int) -> Callable:
        '''
        Returns the Gaussian Process functions for both transition and emission edges.
        
        Parameters:
        -----------
        moment: int
            The moment of interest. Either 0 (for mean) or 1 (for variance).
        
        Returns:
        --------
        Callable
            The Gaussian Process function for both transition and emission edges, which takes in (the time-slice, the transition keys, the emission keys, the values for prediction, and the number of samples) as parameters and outputs the prediction from GP / KDEs.
        '''
        assert moment in [0, 1], "Moment must be either 0 or 1."
        def __get_gp_both(t: int, trans_keys: Sequence, emit_keys: Sequence, sample: hDict, n_samples: int) -> tf.Tensor:
            trans = self.get_gp_trans(moment)(
                t, trans_keys, emit_keys, sample, n_samples
            )
            emit = self.get_gp_emit(moment)(t, trans_keys, emit_keys, sample, n_samples)
            return trans + emit

        return __get_gp_both

    def static(self, moment: int) -> hDict:
        '''
        Returns the static function (usually the initial state, i.e. t=0) for the SEMHat object.
        
        Parameters:
        -----------
        moment: int
            The moment of interest. Either 0 (for mean) or 1 (for variance).
        
        Returns:
        --------
        hDict
            The static function at a certain time-slice of a given moment.
        '''
        assert moment in [0, 1], "Moment must be either 0 or 1."

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

    def dynamic(self, moment: int) -> hDict:
        '''
        Returns the dynamic function for the SEMHat object.
        
        Parameters:
        -----------
        moment: int
            The moment of interest. Either 0 (for mean) or 1 (for variance).
        
        Returns:    
        --------
        hDict
            The dynamic function at a certain time-slice of a given moment.
        '''
        assert moment in [0, 1], "Moment must be either 0 or 1."

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
