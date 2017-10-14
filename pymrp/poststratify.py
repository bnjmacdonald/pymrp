"""Poststratifies data.

"""

import warnings
import numpy as np
import pandas as pd


def poststratify(values, weights, group):
    """poststratifies.

    Notes:

        assumes that group is in the index of values.

    Arguments:

        values (pd.Series): 1d array of cell values to post-stratify.

        weights: 1d array of cell weights to use.

        group: 1d array containing group indicator
    """
    assert values.shape[0] == weights.shape[0]
    values_weighted = np.multiply(values, weights)
    values_gp = values_weighted.groupby(group).sum()
    # values_gp.name = 'values_poststrat'
    # print(values_gp.quantile(q=np.arange(0.1, 1.1, 0.1)))
    if values_gp.isnull().sum() > 0:
        warnings.warn("Poststratification produced missing values.", Warning)
    return values_gp


def get_cell_weights(data, group, margin):
    """constructs a 1d array of cell weights.

    Arguments:
        data (pd.DataFrame): 2d array of individual-level data to use in
            constructing weights.
        group (str or list): str or list of str representing grouping variable
            for weights (e.g. if post-stratifying to construct
            constituency-level estimates, then group='constituency').
        margins (str or list): str or list of str representing margins by
            which to construct weights within the grouping variable. For
            example, if post-stratifying by age and gender to construct
            constituency-level estimates, then use group='constituency' and
            margin=['age', 'gender'].

    Example::

        get_weights(data, group='constituency_pk', margins=['age', 'gender'])

    Todo:
        * may not work when group is a list containing multiple strings. Fix
            this.

    Returns:
        weights: 1d array of cell weights, where weights sum to one for each
            level of the grouping variable (e.g. weights on age and gender
            breakdown sum to one within each constituency). Each weight
            represents the proportion of individuals in that cell out of all
            individuals in that group (e.g. % of all residents in a given
            constituency that are males of age 18-24).

    """
    if isinstance(group, str):
        group = [group]
    if isinstance(margin, str):
        margin = [margin]
    groupby = group + margin
    cell_counts = data[groupby].groupby(groupby).size()
    cell_counts.name = 'cell_count'
    # constit_counts = poststrat_data_indiv.groupby('ke2009a_constit_pk').size()
    # assert all(constit_counts == cell_counts_unstack.sum(axis=0))
    cell_counts_unstack = cell_counts.unstack(group)
    weights_unstack = cell_counts_unstack.divide(cell_counts_unstack.sum(axis=0), axis=1)
    assert all(weights_unstack.sum(axis=0).round(4) == 1.0)
    weights = weights_unstack.stack()
    weights = weights.reorder_levels(groupby).sort_index()
    weights.name = 'cell_prop'
    assert weights.isnull().sum() == 0
    return weights
