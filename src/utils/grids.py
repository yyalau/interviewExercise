from data_struct import hDict
import numpy as np
import tensorflow_probability as tfp

class DistU:
    def __init__(self, dist):
        self.dist = dist
    
    def sample(self, n_samples = 1):
        return self.dist.sample(n_samples).numpy()


def get_interv_sampler(exp_set, limits):
    
    i_sampler = hDict(variables = exp_set, nT = 1, nTrials = 1, )
    
    f= lambda i, es: np.array([limits[var][i] for var in es], dtype=np.float32)
    for es in exp_set:        
        low = f(0, es); high = f(1, es)
        i_sampler[es] = DistU(tfp.distributions.Uniform(low=low, high=high))

    return i_sampler


def get_ndim_igrid(limits: list, size_intervention_grid: int = 100) -> np.array:
    """
    Usage: combine_n_dimensional_intervention_grid([[-2,2],[-5,10]],10)
    """
    if any(isinstance(el, list) for el in limits) is False:
        # We are just passing a single list
        return np.linspace(limits[0], limits[1], size_intervention_grid)[:, None]
    else:
        extrema = np.vstack(limits)
        inputs = [np.linspace(i, j, size_intervention_grid) for i, j in zip(extrema[:, 0], extrema[:, 1])]
        return np.dstack(np.meshgrid(*inputs)).ravel("F").reshape(len(inputs), -1).T


def get_i_grids(exploration_set, intervention_limits, size_intervention_grid=100) -> dict:
    """Builds the n-dimensional interventional grids for the respective exploration sets.

    Parameters
    ----------
    exploration_set : iterable
        All the exploration sets
    intervention_limits : [type]
        The intervention range per canonical manipulative variable in the causal graph.
    size_intervention_grid : int, optional
        The size of the intervention grid (i.e. number of points on the grid)

    Returns
    -------
    dict
        Dict containing all the grids, indexed by the exploration sets
    """

    # Create grids
    intervention_grid = {k: None for k in exploration_set}
    for es in exploration_set:
        if len(es) == 1:
            # Notice that we are splitting by the underscore and this is a hard-coded feature
            intervention_grid[es] = get_ndim_igrid(
                intervention_limits[es[0]], size_intervention_grid
            )
        else:
            if size_intervention_grid >= 100 and len(es) > 2:
                size_intervention_grid = 10

            intervention_grid[es] = get_ndim_igrid(
                [intervention_limits[j] for j in es], size_intervention_grid
            )

    return intervention_grid
