"""implements methods for validating MRP results.
"""

import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns


def get_error(y_true, preds):
    """wrapper to sklean metrics for computing mean squared error and mean
    absolute error.

    Arguments:

        y_true: 1d np.array. Contains true y values.

        preds: pd.DataFrame. Contains predicted values. Each column contains
            an array of predicted values.

    Returns:

        mean_errors: pd.DataFrame. Contains MSE, MAE, mean, and sd of each
            column of predictions in preds.
    """
    mean_errors = []
    for col in preds.columns:
        mse = mean_squared_error(y_true, preds[col])
        mad = mean_absolute_error(y_true, preds[col])
        sd = preds[col].std()
        mean = preds[col].mean()
        mean_errors.append([col, mse, mad, mean, sd])
    mean_errors = pd.DataFrame(mean_errors, columns=['method', 'mse', 'mae', 'mean', 'sd']).set_index('method')
    return mean_errors


def plot_raw_v_mrp(preds, outpath):
    """plots raw vs. MRP group-level estimates on same 2d scatterplot"""
    plt.title('Raw vs. MRP estimates')
    plt.scatter(x=preds.mean_naive, y=preds.mean_mrp, s=preds.n_obs)
    plt.xlabel('Raw estimates')
    plt.ylabel('MRP estimates')
    plt.xlim((0,1.0))
    plt.ylim((0,1.0))
    plt.plot((0, 1.0), (0, 1.0), ls="--", c=".5", linewidth=0.7)
    plt.axhline(y=preds.mean_naive.mean(), ls="--", c=".5", linewidth=0.7)
    fname = 'raw_v_mrp.png'
    plt.savefig(os.path.join(outpath, fname), dpi=200, bbox_inches='tight')
    plt.close()
    return 0


def plot_shrinkage(preds, outpath):
    """Plots 1x3 grid of subplots showing shrinkage in MRP estimates
    relative to raw estimates.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
    ax1.scatter(x=preds.n, y=preds.mean_naive, s=0.7)
    ax2.scatter(x=preds.n, y=preds.mean_ps, s=0.7)
    ax3.scatter(x=preds.n, y=preds.mean_mrp, s=0.7)
    ax1.set_xlabel('# obs.')
    ax2.set_xlabel('# obs.')
    ax3.set_xlabel('# obs.')
    ax1.set_ylabel('Raw estimates')
    ax2.set_ylabel('PS estimates')
    ax3.set_ylabel('MRP estimates')
    ax1.axhline(y=preds.mean_naive.mean(), ls="--", c=".5", linewidth=0.7)
    ax2.axhline(y=preds.mean_naive.mean(), ls="--", c=".5", linewidth=0.7)
    ax3.axhline(y=preds.mean_naive.mean(), ls="--", c=".5", linewidth=0.7)
    fname = 'shrinkage.png'
    fig.set_size_inches(14, 4)
    plt.savefig(os.path.join(outpath, fname), dpi=200, bbox_inches='tight')
    plt.close()


def truth_comparison_plot(data, preds, outcome, group, order, outpath):
    ax = sns.pointplot(x='value', y=group, hue='variable', data=data, ci=None, linestyles=['', '', '', '-'], scale=0.6, orient='h', order=order)
    plt.setp(ax.collections[:3], sizes=preds.set_index(group).loc[order].n)
    plt.legend(title=None, loc='lower right')
    ax.set_xlabel(outcome.replace('_', ' ').title())
    ax.set_ylabel('Constituency')
    ax.get_yaxis().set_ticks([])
    # sns.factorplot(x='value', y='group', hue='variable', data=melted, orient='h')
    # sns.stripplot(x='value', y='group', hue='variable', data=melted)
    fname = 'comparison_{0}'.format(outcome)
    plt.gcf().set_size_inches(8, 10)
    plt.savefig(os.path.join(outpath, fname), dpi=200, bbox_inches='tight')
    plt.close()

