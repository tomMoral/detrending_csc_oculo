# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Tom Dupre La Tour <tom.duprelatour@telecom-paristech.fr>
#          Umut Simsekli <umut.simsekli@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Clement Lalanne <clement.lalanne@ens.fr>

import numpy as np
from scipy import linalg, optimize

from .utils import construct_X, check_consistent_shape

from prox_tv import tv1_1d

def update_trend(X, z_hat, d_hat, reg_trend=0.1, ds_init=None, debug=False,
             solver_kwargs=dict(), sample_weights=None, verbose=0):
    """Learn d's in time domain.

    Parameters
    ----------
    X : array, shape (n_trials, n_times)
        The data for sparse coding
    Z : array, shape (n_atoms, n_trials, n_times - n_times_atom + 1)
        The code for which to learn the atoms
    n_times_atom : int
        The shape of atoms.
    lambd0 : array, shape (n_atoms,) | None
        The init for lambda.
    debug : bool
        If True, check grad.
    solver_kwargs : dict
        Parameters for the solver
    sample_weights: array, shape (n_trials, n_times)
        Weights applied on the cost function.
    verbose : int
        Verbosity level.

    Returns
    -------
    d_hat : array, shape (k, n_times_atom)
        The atom to learn from the data.
    lambd_hats : float
        The dual variables
    """
    conv_part = construct_X(z_hat, d_hat)
    trend = np.zeros(X.shape)
    to_analyse = X-conv_part

    for i in range(X.shape[0]):
        trend[i] = tv1_1d(to_analyse[i], reg_trend)
    
    return trend
    