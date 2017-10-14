"""Constructs estimates of group-level preferences for binomial outcome data
using a multilevel logistic regression model.

Usage:

    Example 1::

        python citizen_priorities.py --outcome party_narc --group constituency_pk --lvl1_formula "urban + C(age_bin) + female" --lvl2_formula "-1 + C(ke2009a_prov) + yrschool" --iter 500 --chains 2 --model_fname binomial_mlm
"""

import sys
import argparse

import os
import re
import warnings
import numpy as np
import pandas as pd
import patsy
from pprint import pprint
from collections import OrderedDict
# from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, classification_report, accuracy_score
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

import settings
import utils
import mrp
from plot import PlotConfig

plot_config = PlotConfig()

def parse_args():
    parser = argparse.ArgumentParser(add_help=True, description='Estimates multilevel logistic regression model.')
    parser.add_argument(
        '-o',
        '--outcome',
        type=str,
        required=True,
        help='Outcome variable (e.g. "--outcome mp_issue_responsibility_any_control_crime").'
    )
    parser.add_argument(
        '-g',
        '--group',
        nargs='+',
        required=True,
        help='Grouping variable(s) (e.g. "--group constituency_pk").'
    )
    parser.add_argument(
        '-l1',
        '--lvl1_formula',
        type=str,
        required=True,
        help='Right-hand side of formula for level 1 of multilevel model (e.g. "-1 + C(age_bin) + female").'
    )
    parser.add_argument(
        '-l2',
        '--lvl2_formula',
        type=str,
        required=True,
        help='Right-hand side of formula for level 2 of multilevel model (e.g. "-1 + C(ke2009a_prov) + yrschool").'
    )
    parser.add_argument(
        '-m',
        '--model_fname',
        type=str,
        required=True,
        help='Model filename (e.g. "binomial_mlm").'
    )
    parser.add_argument(
        '-i',
        '--iter',
        type=int,
        required=False,
        default=2000,
        help='Number of draws to take from posterior distribution in each chain.'
    )
    parser.add_argument(
        '-c',
        '--chains',
        type=int,
        required=False,
        default=4,
        help='Number of chains.'
    )
    # parser.add_argument(
    #     '--margin',
    #     nargs='+',
    #     required=True,
    #     help='Margin variable(s) (e.g. "--margin age_bin female").'
    # )
    # parser.add_argument(
    #     '--lvl2_predictors',
    #     nargs='+',
    #     required=True,
    #     help='Group-level predictor(s) (e.g. "--lvl2_predictors yrschool").'
    # )
    # args = parser.parse_args()
    args, unparsed = parser.parse_known_args()
    sys.argv = [sys.argv[0]]  # kludge.
    return args

args = parse_args()


def custom_formatwarning(msg, category, filename, lineno, *a):
    # ignore everything except the message
    return '(' + str(filename) + ') ' + str(category) + ': ' + str(msg) + ' Line: ' + str(lineno) + '\n'

warnings.formatwarning = custom_formatwarning


if __name__ == '__main__':
    # class args(object):
    #     outcome = 'is_employed'  # 'mp_issue_responsibility_any_control_crime'
    #     group = ['constituency_pk']
    #     lvl1_formula = "-1 + C(age_bin) + female"
    #     lvl2_formula = "-1 + C(ke2009a_prov) + yrschool"
    #     iter = 500
    #     chains = 2
    #     model_fname = 'binomial_mlm'



# 1. reads in individual-level response data (i.e. survey data).
afro_filename = 'afro_clean.csv'
data_afro = pd.read_csv(os.path.join(settings.DATA_DIR, 'clean', afro_filename))
# pprint.pprint(sorted(data_afro.columns.tolist()))

# 2. reads in poststratification data (i.e. census data).
census_filename = 'census_clean.csv'
data_census = pd.read_csv(os.path.join(settings.DATA_DIR, 'clean', census_filename))

# harmonizes variable names as needed.
data_census.rename(columns={'sex': 'female'}, inplace=True)
data_census.rename(columns={'ke2009a_urban': 'urban'}, inplace=True)
data_census.rename(columns={'educke': 'educ_level'}, inplace=True)

# 3. Creates dataset of geographic-level predictors.

# Todo: aggregate census and afrobarometer data to group-level.

# NOTE: kludge for creating population size of group. Should have a separate file that aggregates the census data to the constituency-level, as there will be other group-level only variables I'll construct later (e.g. square km).
group_population = data_census.groupby(args.group).size()
group_population.name = 'population'
data_census = pd.merge(data_census, group_population.to_frame(), left_on=args.group, right_index=True, how='left')
assert all(data_census.groupby(args.group)['population'].nunique() == 1)


class BinomialMRP(object):

    def __init__(self, verbose):
        self.verbose = verbose
        lvl1_model_termlist = patsy.ModelDesc.from_formula(args.lvl1_formula).rhs_termlist
        lvl1_predictors = list(set([re.sub(r'^C\(|\)$', '', term.name()) for term in lvl1_model_termlist if ':' not in term.name() and term.name() != 'Intercept']))
        lvl2_model_termlist = patsy.ModelDesc.from_formula(args.lvl2_formula).rhs_termlist
        lvl2_predictors = list(set([re.sub(r'^C\(|\)$', '', term.name()) for term in lvl2_model_termlist if ':' not in term.name() and term.name() != 'Intercept']))
        assert len(set(lvl1_predictors).intersection(lvl2_predictors)) == 0, 'cannot include the same variable in both model levels.'
        if self.verbose > 1:
            print('Level 1 predictors: ', lvl1_predictors)
            print('Level 2 predictors: ', lvl2_predictors)

    def fit(self):
        """fits logistic multilevel model for an individual survey response given:
    # 4. Fit a regression model for an individual survey response given:
    # (a) individual-level covariates that also appear in the poststratification 
    #   dataset; and 
    # (b) group-level covariates.
    """

    # keeps only complete cases.
    print('Dataframe sizes before dropping NA:')
    print(data_afro.shape)
    print(data_census.shape)
    data_afro = data_afro[[args.outcome] + args.group + lvl1_predictors].dropna(axis=0)
    data_census = data_census[args.group + lvl1_predictors + lvl2_predictors].dropna(axis=0)
    print('Dataframe sizes after dropping NA:')
    print(data_afro.shape)
    print(data_census.shape)

    print(data_afro[args.outcome].value_counts())

    # converts group and lvl1_predictors columns to integer if they are floats.
    for col in args.group + lvl1_predictors:
        if data_afro[col].dtype in [np.float32, np.float64]:
            data_afro[col] = data_afro[col].astype(np.int64)
        if data_census[col].dtype in [np.float32, np.float64]:
            data_census[col] = data_census[col].astype(np.int64)

    # converts outcome to integer if it is a float.
    # NOTE: this makes sense because all models must be logit models at the moment.
    if data_afro[args.outcome].dtype not in [np.int32, np.int64]:
        warnings.warn('{0} was in {1} format and had to be converted to np.int64.'.format(args.outcome, data_afro[args.outcome].dtype), Warning)
        print('Old distribution: ')
        print(data_afro[args.outcome].value_counts(dropna=False))
        data_afro[args.outcome] = data_afro[args.outcome].astype(np.int64)
        print('New distribution: ')
        print(data_afro[args.outcome].value_counts(dropna=False))

    # creates 'group' variable for simplicity.
    data_census['group'] = utils.concat_columns(data_census, args.group, sep='_')
    data_afro['group'] = utils.concat_columns(data_afro, args.group, sep='_')
    print('Number of unique groupings:')
    print('Census: {0}'.format(data_census['group'].nunique()))
    print('Survey: {0}'.format(data_afro['group'].nunique()))

    # converts group to categorical so that there are no "jumps" in the group index.
    data_afro['group_code'] = data_afro['group'].astype('category').cat.codes.astype(np.int64).replace('-1', np.nan) + 1
    group_code_dict = dict(zip(data_afro['group_code'].tolist(), data_afro['group'].tolist()))  # dict of cat code -> constituency pk
    group_code_dict_rev = {v: k for k, v in group_code_dict.items()}
    assert data_afro['group_code'].nunique() == data_afro['group'].nunique()
    assert len(group_code_dict) == len(group_code_dict_rev)

    # computes means of outcome columns.
    # issue_cols = [col for col in data_afro.columns if col.startswith('mp_issue_responsibility_any_')]
    # print(data_afro[issue_cols].mean().sort_values(ascending=False))
    # constit_counts = data_afro.groupby('group')[args.outcome].sum()
    group_means = data_afro.groupby('group')[args.outcome].mean()
    # print(unemp.quantile(q=np.arange(0, 1.1, 0.1)))
    mean = group_means.mean()
    std = group_means.std()
    print('Descriptives of outcome:')
    print(data_afro[args.outcome].mean())
    print(mean)
    print(std)

    # constructs design matrix (X)
    # X = []
    # for col in lvl1_predictors:
    #     if data_afro[col].nunique() < 2:
    #         raise RuntimeError('All --lvl1_predictors columns must take on at least two distinct values.')
    #     elif data_afro[col].nunique() == 2:
    #         dummies = data_afro[col]
    #     elif data_afro[col].nunique() > 2:
    #         dummies = pd.get_dummies(data_afro[col], prefix=col, prefix_sep='_')
    #     X.append(dummies)
    # X = pd.concat(X, axis=1)

    # adds interactions.
    # args.lvl1_formula = '-1 + C(age_bin) + female'
    dm = patsy.dmatrix(args.lvl1_formula, data=data_afro)
    # [re.search(r'', term) for term in dm.design_info.term_names]
    # np.asarray(dm)
    X = pd.DataFrame(dm, columns=dm.design_info.column_names)
    pprint(X.columns.tolist())

    # poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    # X2 = poly.fit_transform(X)
    # feature_names = poly.get_feature_names()

    # constructs group-level design matrix (U)
    # U = []
    # for col in lvl2_predictors:
    #     if data_census[col].nunique() < 2:
    #         raise RuntimeError('All --lvl1_predictors columns must take on at least two distinct values.')
    #     elif data_census[col].nunique() == 2:
    #         dummies = data_census[col]
    #     elif data_census[col].nunique() > 2:
    #         dummies = pd.get_dummies(data_census[col], prefix=col, prefix_sep='_')
    #     U.append(dummies)
    # print(lvl2_predictors)
    # print(data_census.head())

    # formula = 
    dm_census = patsy.dmatrix(args.lvl2_formula, data=data_census)
    U_temp = pd.DataFrame(dm_census, index=data_census.group, columns=dm_census.design_info.column_names)
    U = U_temp.groupby(level='group').mean()
    U = U[U.index.isin(data_afro['group'])]
    U.index = U.index.map(lambda x: group_code_dict_rev[x])
    print(U.head())
    print(U['C(ke2009a_prov)[nyanza]'].quantile(q=np.arange(0,1.1,0.1)))
    U_long = data_afro.set_index('group_code').join(U, how='left')[U.columns]
    # U['intercept'] = np.ones((U.shape[0],), dtype=np.int64)
    # U = U[['intercept'] + lvl2_predictors]
    assert U_long.shape[0] == X.shape[0] and X.shape[0] == data_afro.shape[0]
    assert all(X.isnull().sum() == 0) and all(U_long.isnull().sum() == 0)

    # creates a cell dataframe
    # want grid of cells with group values at which to make predictions.
    grid = OrderedDict(
        [('group_code', data_afro['group_code'].unique())] +
        [(col, data_afro[col].unique()) for col in lvl1_predictors]
    )
    grid = utils.expand_grid(grid)

    # age_levels = data_afro['age_bin'].unique()
    # formula = '-1 + C(group_code)'
    # formula2 = '-1 + urban + C(age_bin) + female'
    # dm = patsy.dmatrix(formula, data=grid)
    dm = patsy.dmatrix(args.lvl1_formula, data=grid)
    # np.asarray(dm)
    # a_grid = pd.DataFrame(dm, columns=dm.design_info.column_names)
    group_grid = grid.group_code
    X_grid = pd.DataFrame(dm, columns=dm.design_info.column_names)
    # grid['group'] = grid['group_code'].apply(lambda x: group_code_dict[x])
    U_grid = grid.set_index('group_code').join(U, how='left')[U.columns]
    assert all(X.columns == X_grid.columns)
    assert all(U_long.columns == U_grid.columns)
    assert group_grid.shape[0] == X_grid.shape[0] and group_grid.shape[0] == U_grid.shape[0]
    assert X.shape[1] == X_grid.shape[1]  and U.shape[1] == U_grid.shape[1]
    
    # grid_wide = pd.concat([a_grid, X_grid, U_grid], axis=1)
    # print(grid_wide.head())

    stan_data = {
        # 'sigma_a': std,
        'K': X.shape[1],
        'M': U_long.shape[1],
        'N': data_afro.shape[0],
        'J': data_afro['group'].nunique(),
        'group': data_afro['group_code'].values,
        'y': data_afro[args.outcome].values,
        'X': X.values,
        'U': U_long.values,
        # value grid for predictions
        'N2': group_grid.shape[0],
        'group_new': group_grid.values,
        'X_new': X_grid.values,
        'U_new': U_grid.values,
    }

    # Todo: decide on reasonable prior for the alphas

    print('Level 1 design matrix:')
    print(X.shape)
    print(X.head())
    print('Level 2 design matrix: ')
    print(U_long.shape)
    print(U_long.head())
    
    # fits the model
    stan_fit = mrp.fit(fname=args.model_fname + '.pkl', file=args.model_fname + '.stan', model_name=args.model_fname, data=stan_data, iter=args.iter, chains=args.chains, n_jobs=4)  # init=0
    # print(stan_fit)
    # fit.traceplot()
    # plt.show()
    
    # extracts predicted values.
    pred_draws = pd.DataFrame(stan_fit.extract('y_pred', permuted=True)['y_pred'])
    pred_means = pred_draws.mean(axis=0)
    pred_sd = pred_draws.std(axis=0)

    # simple evaluation
    y_logit_draws = pd.DataFrame(stan_fit.extract('y_logit', permuted=True)['y_logit'])
    y_means = utils.sigmoid(y_logit_draws.mean(axis=0))
    
    r2 = r2_score(data_afro[args.outcome].values, y_means)
    y_preds = (y_means > 0.5).astype(int)
    print(classification_report(data_afro[args.outcome].values, y_preds))
    print('Accuracy:')
    print(accuracy_score(data_afro[args.outcome].values, y_preds))
    print('R-squared:')
    print(r2)

    # make a dataframe of parameter estimates for all chains
    group_intercept_draws = pd.DataFrame.from_records(stan_fit.extract('a', permuted=True)['a'])
    group_intercept_draws.columns = ['a{0}'.format(col + 1) for col in group_intercept_draws.columns]
    
    coef_draws = pd.DataFrame(stan_fit.extract('b', permuted=True)['b'])
    coef_draws.columns = X.columns

    group_coef_draws = pd.DataFrame(stan_fit.extract('z', permuted=True)['z'])
    group_coef_draws.columns = U.columns

    intercepts = group_intercept_draws.describe().transpose()
    coefs = coef_draws.describe().transpose()
    group_coefs = group_coef_draws.describe().transpose()

    print(intercepts)
    print(coefs)
    print(group_coefs)
    # intercepts.index = data_afro['group_code'].values
    # intercepts = intercepts.reset_index().drop_duplicates(subset='index').set_index('index')
    # intercepts.sort_index(inplace=True)
    # intercepts.index = ['a{0}'.format(i) for i in intercepts.index]
    # assert intercepts.shape[0] == data_afro['group'].nunique()
    # params = pd.concat([intercepts, coef_draws.mean(axis=0)], axis=0).squeeze()

    # params = pd.concat([, pd.DataFrame(stan_fit.extract('b', permuted=True)['b'])], axis=1)
    # columns=['a{0}'.format(i+1) for i in range(data_afro.pk_codes.nunique())]

    # get the predicted values for each chain. This is super convenient in pandas because
    # it is possible to have a single column where each element is a list
    # chainPreds = params.apply(stanPred, axis = 1)


    # grid_wide = pd.concat([pd.get_dummies(grid['group_code'], prefix='a', prefix_sep=''), pd.get_dummies(grid['age_bin'], prefix='b', prefix_sep=''), grid], axis=1)
    # grid_wide.drop(['group_code', 'age_bin'], axis=1, inplace=True)

    # NOTE: must be careful to make sure these align properly.
    # print(params.index)
    # print(grid_wide.columns)
    # assert params.shape[0] == grid_wide.shape[1]
    # # assert all(params.index == grid_wide.columns)

    # # makes prediction for each cell from multilevel model.
    # preds = mrp.predict(params, grid_wide, link=utils.sigmoid)
    # preds = pd.Series(preds, name=args.outcome)  # .replace('mp_issue_responsibility_any_', '')
    print(pred_means.quantile(q=np.arange(0, 1.1, 0.1)))
    assert pred_means.max() <= 1.0 and pred_means.min() >= 0.0

    # 
    grid['group'] = grid['group_code'].apply(lambda x: group_code_dict[x])
    grid.drop('group_code', axis=1, inplace=True)
    grid = grid[['group'] + lvl1_predictors]  # reorders columns
    grid_index = pd.MultiIndex.from_tuples(grid.values.tolist(), names=grid.columns)
    preds = pred_means.to_frame().set_index(grid_index).squeeze()
    assert preds.isnull().sum() == 0


    # 5. Post-stratify.

    # creates cell weights.
    # print(sorted(data_census['constituency_pk'].unique()))
    # print(sorted(preds.index.get_level_values('constituency_pk').unique()))
    weights = mrp.get_weights(data_census, group='group', margin=lvl1_predictors)
    weights = weights.loc[grid_index]
    if weights.isnull().sum() > 0:
        warnings.warn('{0} weights are NaN'.format(weights.isnull().sum()), Warning)
        utils.print_full(weights[weights.isnull()])
        weights.fillna(value=0.0, inplace=True)
    print(weights.quantile(q=np.arange(0.0, 1.1, 0.1)))

    # aligns indices between preds and weights.
    preds = preds.loc[weights.index]
    assert preds.shape == weights.shape and all(preds.index == weights.index)

    # poststratifies (i.e. adjusts predictions using cell weights).
    priorities_mrp = mrp.poststratify(preds, weights, group='group')
    priorities_mrp.name = 'mean_mrp'

    # computes naive means
    priorities_means_naive = data_afro.groupby('group')[args.outcome].mean()
    priorities_means_naive.name = 'mean_naive'
    assert priorities_means_naive.isnull().sum() == 0

    # poststratifies on naive means.
    cell_means = data_afro.groupby(['group'] + lvl1_predictors)[args.outcome].mean()
    cell_means = cell_means.loc[weights.index]
    priorities_means_ps = mrp.poststratify(cell_means, weights, group='group')
    priorities_means_ps.name = 'mean_ps'
    assert priorities_means_ps.isnull().sum() == 0

    # disagg_means = data_afro.groupby(args.group)[args.outcome].mean()
    # disagg_means.name = 'preds_disagg'

    # combines three ways of estimating group priorities.
    priorities = pd.concat([priorities_mrp, priorities_means_ps, priorities_means_naive], axis=1)
    n_obs = data_afro.groupby('group')[args.outcome].size()
    n_obs.name = 'n_obs'
    priorities = priorities.join(n_obs)

    print(priorities.mean(axis=0))

    # preds_merged = pd.merge(preds.to_frame(), disagg_means.to_frame(), left_index=True, right_index=True)
    # preds_merged = pd.merge(preds_merged, disagg_n_obs.to_frame(), left_index=True, right_index=True)
    # assert preds_merged.n_obs.sum() == data_afro.shape[0]


    # EVALUATION
    # ----------
    
    # marginal means.
    print(preds.head())
    for col in lvl1_predictors:
        print(preds.groupby(level=col).mean())
    # print(preds.groupby(level=['female', 'age_bin']).mean())
    # print(preds.groupby(level=['urban', 'age_bin']).mean())

    # basic model checking
    # todo...

    # plots raw vs. MRP estimates on same 2d scatterplot
    plt.title('Raw vs. MRP estimates')
    plt.scatter(x=priorities.mean_naive, y=priorities.mean_mrp, s=priorities.n_obs)
    plt.xlabel('Raw estimates')
    plt.ylabel('MRP estimates')
    plt.xlim((0,1.0))
    plt.ylim((0,1.0))
    plt.plot((0, 1.0), (0, 1.0), ls="--", c=".5", linewidth=0.7)
    plt.axhline(y=priorities.mean_naive.mean(), ls="--", c=".5", linewidth=0.7)
    fname = 'preds_mrp_v_disagg'
    plt.savefig(os.path.join(settings.OUTPUT_DIR, 'figures', fname), dpi=plot_config.dpi, bbox_inches='tight')
    plt.close()

    # Creates 1x3 grid of subplots showing shrinkage in MRP estimates relative to raw estimates.
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
    ax1.scatter(x=priorities.n_obs, y=priorities.mean_naive, s=0.7)
    ax2.scatter(x=priorities.n_obs, y=priorities.mean_ps, s=0.7)
    ax3.scatter(x=priorities.n_obs, y=priorities.mean_mrp, s=0.7)
    ax1.set_xlabel('# obs.')
    ax2.set_xlabel('# obs.')
    ax3.set_xlabel('# obs.')
    ax1.set_ylabel('Raw estimates')
    ax2.set_ylabel('PS estimates')
    ax3.set_ylabel('MRP estimates')
    ax1.axhline(y=priorities.mean_naive.mean(), ls="--", c=".5", linewidth=0.7)
    ax2.axhline(y=priorities.mean_naive.mean(), ls="--", c=".5", linewidth=0.7)
    ax3.axhline(y=priorities.mean_naive.mean(), ls="--", c=".5", linewidth=0.7)
    fname = 'preds_comparison'
    fig.set_size_inches(14, 4)
    plt.savefig(os.path.join(settings.OUTPUT_DIR, 'figures', fname), dpi=plot_config.dpi, bbox_inches='tight')
    plt.close()

    # saves poststratification estimates to file.
    fname = '{0}_priorities_{1}.csv'.format('_'.join(args.group), args.outcome)
    priorities.to_csv(os.path.join(settings.DATA_DIR, 'clean', fname), index=True, header=True)

