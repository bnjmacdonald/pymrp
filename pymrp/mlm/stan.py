"""implements methods for estimating Bayesian multi-level models using Stan.
"""

import re
import pickle
import numpy as np
import pystan


def fit(path, init_kws, **kwargs):
    """fits a multilevel model using Stan.

    Arguments:

        path: str. Path to existing fitted Stan model. If no model exists, a new
            one is created.

        init_kws: dict. Other keywords arguments to pass to pystan.StanModel if
             a new model is created (e.g. {"model_name": "my_stan_model"}).

        **kwargs: keyword arguments to pass to pystan.StanModel.sampling.

    Returns:

        fit: fitted pystan model.
    """
    try:
        with open(path, 'rb') as f:
            sm = pickle.load(f)
    except IOError:
        print('Could not load existing model. Compiling model from scratch.')
        sm = pystan.StanModel(file=path, **init_kws)
        with open(path, 'wb') as f:
            pickle.dump(sm, f, protocol=2)
    kwargs.pop('file')
    kwargs.pop('model_name')
    fit = sm.sampling(**kwargs)
    return fit


def predict(params, grid, link):
    """predicts values from params based on values in grid.

    Arguments:
        params: 1d array of parameters (e.g. coefficients from multilevel
            model).
        grid: 2d array of values to make predictions at. Each row is a
            combination of values, whose order must line up with params.
        link: callable link function (e.g. sigmoid link). Default: None (
            representing linear link).

    Returns:
        preds: 1d array of predictions.
    """
    preds = np.matmul(grid, params)
    if link is not None:
        preds = link(preds)
    # assert utils.sigmoid(np.dot(new_grid.loc[0], params)) == preds[0]
    return preds
