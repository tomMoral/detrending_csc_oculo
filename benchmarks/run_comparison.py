import os
import itertools
import numpy as np
import prox_tv as tv
from datetime import datetime
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from joblib import Memory

from alphacsc_trend.datasets import oculo
from alphacsc_trend.utils import check_random_state
from alphacsc_trend.learn_d_z_multi import learn_d_z_multi
from alphacsc_trend.loss_and_gradient import construct_X_multi
from alphacsc_trend.utils.signal import check_univariate_signal

NO_DETREND = 1000000
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')


mem = Memory(location='.', verbose=0)


def evaluate_d_hat(patterns, d_hat):
    n_atoms = len(d_hat)
    corr = np.zeros((len(patterns), n_atoms))
    for i, pat in enumerate(patterns):
        l_pat = pat.nonzero()[0].max()
        pat = pat[:l_pat+1]
        pat -= pat.mean()
        pat /= np.linalg.norm(pat)
        pat = np.r_[pat, pat]
        for k, d_k in enumerate(d_hat):
            d_k -= d_k.mean()
            d_k /= np.linalg.norm(d_k)
            corr[i, k] = max(*[abs(np.correlate(d_k, pat[t:t+l_pat])).max()
                               for t in range(l_pat)])
    return corr


def r2(x_true, x_pred):
    x_true, x_pred = x_true.ravel(), x_pred.ravel()
    return np.linalg.norm(x_true - x_pred) / np.linalg.norm(x_true)


def get_trend_init(X, trend_reg):

    trend_init = np.zeros_like(X)
    for i in range(X.shape[0]):
        trend_init[i] = tv.tv1_1d(X[i], trend_reg)
    return trend_init


def get_lambda_max_tv(X):
    return abs(np.diff(X)).max()


@mem.cache
def run_one(X_i, nyst_i, pattern_i, csc_params, trend_reg, nyst_reg,
            random_state, i=0, display=False):

    l_pat = pattern_i[0].nonzero()[0].max()
    n_times_atom = int(l_pat * 1.5)

    trend_reg_ = trend_reg * get_lambda_max_tv(X_i)
    trend_init = get_trend_init(X_i, trend_reg_)

    _, _, d_hat_detrend_init, z_hat_detrend_init, *_ = learn_d_z_multi(
        X_i - trend_init, n_times_atom=n_times_atom, reg=nyst_reg,
        trend_reg=NO_DETREND, random_state=random_state,
        **csc_params
    )
    _, _, d_hat, z_hat, trend_hat, *_ = learn_d_z_multi(
        X_i, n_times_atom=n_times_atom, reg=nyst_reg, trend_reg=trend_reg_,
        random_state=random_state, **csc_params
    )

    # Remove unused channel for evaluation and plots
    X_hat = construct_X_multi(z_hat, d_hat)[0, 0]
    X_hat_detrend_init = construct_X_multi(z_hat_detrend_init,
                                           d_hat_detrend_init)[0, 0]
    xi = X_i[0, 0]
    d_hat = d_hat[:, 0]
    d_hat_detrend_init = d_hat_detrend_init[:, 0]
    trend_hat = trend_hat[0, 0]
    trend_init = trend_init[0, 0]

    # Compute the metrics
    res_trial = dict(
        r2_full=r2(nyst_i, X_hat),
        r2_init=r2(nyst_i, X_hat_detrend_init),
        r2_detrend_init=r2(nyst_i, xi - trend_init),
        r2_detrend_hat=r2(nyst_i, xi - trend_hat),
        corr_full=evaluate_d_hat(pattern_i, d_hat),
        corr_init=evaluate_d_hat(pattern_i, d_hat_detrend_init),
        trend_reg=trend_reg, nyst_reg=nyst_reg,

    )

    print("=" * 80)
    print(f"Trial {i} (l1: {nyst_reg}, tv: {trend_reg})")
    print("=" * 80)
    print(f"R2 init: {res_trial['r2_detrend_init']}")
    print(f"R2 detrend: {res_trial['r2_detrend_hat']}")
    print(f"R2 denoise: {res_trial['r2_full']}")
    print(f"R2 denoise init: {res_trial['r2_init']}")
    print(f"Corr D_hat: {res_trial['corr_full']}")
    print(f"Corr D_hat init: {res_trial['corr_init']}")
    print("=" * 80)

    if display:
        plt.subplot(1, 2, 1)
        plt.plot(xi)
        plt.plot(trend_hat)

        plt.subplot(1, 2, 2)
        plt.plot(nyst_i)
        plt.plot(xi - trend_hat)

        plt.figure()
        plt.plot(d_hat.T)
        plt.plot(pattern_i.T)
        plt.show()

        import IPython
        IPython.embed(colors='neutral')

    return res_trial


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description="Run CSC detrending on simulated oculographic signals")
    parser.add_argument('--debug', action='store_true',
                        help="Use the debug mode")
    parser.add_argument('--n-jobs', type=int, default=1,
                        help="# of jobs to run the benchmark")
    parser.add_argument('--n-trials', type=int, default=100,
                        help="# of run to perform")
    args = parser.parse_args()

    n_jobs = 1 if args.debug else args.n_jobs

    list_nyst_reg = [.9, .8, .5, .1]
    list_trend_reg = [0.05, .1, .2, .5, .9, NO_DETREND]

    n_times = 10000
    n_trials = 1 if args.debug else args.n_trials
    n_iter = 150 if args.debug else 200

    verbose = 1 if args.debug else 0
    random_state = None if args.debug else 42

    rng = check_random_state(random_state)

    # Algo parameters
    csc_params = dict(
        n_atoms=1, D_init='chunk', rank1=False, lmbd_max='scaled',
        window=True, solver_d_kwargs=dict(max_iter=50, eps=1e-8),
        raise_on_increase=False, n_iter=n_iter, eps=1e-8,
        n_jobs=1, verbose=verbose,
        solver_z='lgcd', solver_z_kwargs=dict(tol=1e-5)
        # algorithm='greedy'
    )

    trend, nyst, patterns = oculo.load_data(
        n_trials=n_trials, n_times=n_times, random_state=rng.randint(int(1e4)))

    X_full = trend + nyst
    X_full = check_univariate_signal(X_full, normalize=False)

    seeds = rng.randint(int(1e5), size=n_trials)
    args_iterator = itertools.product(
        list_nyst_reg, list_trend_reg, zip(X_full, nyst, patterns, seeds)
    )
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_one)(X_i=X_i[None], nyst_i=nyst_i,
                         pattern_i=pattern_i[None],
                         csc_params=csc_params, trend_reg=trend_reg,
                         nyst_reg=nyst_reg, random_state=random_seed, i=i,
                         display=args.debug)
        for i, (nyst_reg, trend_reg,
                (X_i, nyst_i, pattern_i, random_seed)
                ) in enumerate(args_iterator)
    )

    tag = f"{datetime.now().strftime('%Y-%m-%d_%Hh%M')}"
    out_file_name = os.path.join(OUTPUT_DIR, f"results_{tag}.pkl")
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_pickle(out_file_name)
    except Exception:
        import IPython
        IPython.embed(colors='neutral')
