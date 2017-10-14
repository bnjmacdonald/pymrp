"""implements methods for estimating Bayesian multi-level models using stanarm.
"""

import os
import numpy as np
import rpy2.robjects.packages as rpackages
from rpy2.robjects import r, pandas2ri
from sklearn.metrics import r2_score, classification_report, accuracy_score

from pymrp.mlm.utils import get_terms

pandas2ri.activate()
rpackages.importr('rstanarm')

def fit(data, outpath=None, verbosity=0, **kwargs):
    """estimates a multilevel model using the stanarm package in R.

    Todos:

        TODO: before converting data to r_data, filter out columns not appearing
            in formula.
    """
    r_data = pandas2ri.py2ri(data)
    kwargs['data'] = r_data
    kwargs['na.action'] = 'na.omit'
    fit = r.stan_glmer(**kwargs)  # TODO: select appropriate prior.
    if verbosity:
        print(fit)
        # print(fit.rx2('linear.predictors'))
        probs = np.array(fit.rx2('fitted.values'))
        preds = (probs > 0.5).astype(int)
        y = np.array(fit.rx2('y'))
        print('Number of observations: {0}'.format(y.shape[0]))
        print('Distribution of y:\n{0}'.format(np.bincount(y)))
        print('Classification report:')
        print(classification_report(y, preds))
        print('R2 Score:\n', r2_score(y, probs))
        print('Accuracy:\n', accuracy_score(y, preds))
        # r.X11()
        # r.plot(fit)
        # r.posterior_interval(fit, prob=0.95, pars='urban')
        # model evaluation
        # y_draws = r.posterior_predict(fit)
        # preds = r.predict(fit)
        # np.array(r['as.matrix'](fit.rx2('x'))).shape  # design matrix
        # np.array(r['as.matrix'](fit, pars='urban')).shape  # posterior parameter draws
    if verbosity > 1:
        print(r.summary(fit))
    if verbosity > 2:
        inspect(fit)
    if outpath is not None:
        outcome, _, _, _ = get_terms(fit.rx2('formula')[0])
        fname = 'mlm_{0}.rds'.format(outcome)
        save(fit, os.path.join(outpath, fname))
    return fit

def inspect(fit):
    r.launch_shinystan(fit)

def predict(fit, **kwargs):
    """predicts values from fitted stanarm object."""
    pp_draws = np.array(r.posterior_predict(fit, **kwargs))
    return pp_draws.mean(axis=0)

def save(fit, outpath):
    r.saveRDS(fit, outpath)
    return 0

def load(inpath):
    return r.readRDS(inpath)
