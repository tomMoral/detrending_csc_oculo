import numpy as np
import prox_tv as tv
import matplotlib.pyplot as plt
from alphacsc_trend.datasets import oculo

from alphacsc_trend.utils import check_random_state
from alphacsc_trend.learn_d_z_multi import learn_d_z_multi
from alphacsc_trend.loss_and_gradient import construct_X_multi
from alphacsc_trend.utils.signal import check_univariate_signal


DEBUG = False

n_trials = 100
reg_z = .5
trend_reg = .1
no_detrend_reg = 1000000

rng = check_random_state(42)

# Algo parameters
csc_params = dict(
    n_atoms=1, D_init='chunk', rank1=False, lmbd_max='scaled',
    window=True, solver_d_kwargs=dict(max_iter=50, eps=1e-8),
    raise_on_increase=False, n_iter=200, eps=1e-8,
    verbose=1
    # algorithm='greedy'
)


trend, nyst, nyst_patterns = oculo.load_data(n_trials=n_trials, n_times=10000,
                                             random_state=rng.randint(int(1e4)))

X_full = trend + nyst
X_full = check_univariate_signal(X_full, normalize=False)


def evaluate_d_hat(nyst_patterns, d_hat):
    n_atoms = len(d_hat)
    corr = np.zeros((len(nyst_patterns), n_atoms))
    for i, pat in enumerate(nyst_patterns):
        l_pat = pat.nonzero()[0].max()
        pat = nyst_patterns[0, :l_pat]
        pat -= pat.mean()
        pat /= np.linalg.norm(pat)
        for k, d_k in enumerate(d_hat):
            d_k -= d_k.mean()
            d_k /= np.linalg.norm(d_k)
            corr[i, k] = abs(np.correlate(d_k, pat)).max()
    return corr


def r2(x_true, x_pred):
    x_true, x_pred = x_true.ravel(), x_pred.ravel()
    return np.linalg.norm(x_true - x_pred) / np.linalg.norm(x_true)


def get_trend_init(X, trend_reg):

    trend_init = np.zeros_like(X)
    for i in range(X.shape[0]):
        trend_init[i] = tv.tv1_1d(X[i], trend_reg)
    return trend_init


results = []
for i in range(n_trials):
    X = X_full[i:i+1]
    nyst_i = nyst[i]
    pattern = nyst_patterns[i:i+1]

    l_pat = pattern[0].nonzero()[0].max()
    n_times_atom = l_pat + 100

    random_state = rng.randint(int(1e5))
    trend_init = get_trend_init(X, trend_reg)

    _, _, d_hat_no_detrend, z_hat_no_detrend, *_ = learn_d_z_multi(
        X, n_times_atom=n_times_atom, reg=reg_z, trend_reg=no_detrend_reg,
        random_state=random_state, **csc_params
    )

    _, _, d_hat_detrend_init, z_hat_detrend_init, *_ = learn_d_z_multi(
        X - trend_init, n_times_atom=n_times_atom, reg=reg_z,
        trend_reg=no_detrend_reg, random_state=random_state,
        **csc_params
    )
    _, _, d_hat, z_hat, trend_hat, *_ = learn_d_z_multi(
        X, n_times_atom=n_times_atom, reg=reg_z, trend_reg=trend_reg,
        random_state=random_state, **csc_params
    )

    # Remove unused channel for evaluation and plots
    X_hat = construct_X_multi(z_hat, d_hat)[0, 0]
    X_hat_no_detrend = construct_X_multi(z_hat_no_detrend,
                                         d_hat_no_detrend)[0, 0]
    X_hat_detrend_init = construct_X_multi(z_hat_detrend_init,
                                           d_hat_detrend_init)[0, 0]
    X = X[0, 0]
    d_hat = d_hat[:, 0]
    d_hat_no_detrend = d_hat_no_detrend[:, 0]
    d_hat_detrend_init = d_hat_detrend_init[:, 0]
    trend_hat = trend_hat[0, 0]
    trend_init = trend_init[0, 0]

    # Compute the metrics
    res_trial = dict(
        r2_full=r2(nyst_i, X_hat),
        r2_init=r2(nyst_i, X_hat_detrend_init),
        r2_no=r2(nyst_i, X_hat_no_detrend),
        corr_full=evaluate_d_hat(pattern, d_hat),
        corr_init=evaluate_d_hat(pattern, d_hat_detrend_init),
        corr_no=evaluate_d_hat(pattern, d_hat_no_detrend),
    )
    results.append(res_trial)

    print("=" * 80)
    print("Trial", i)
    print("=" * 80)
    print(f"R2 init: {r2(nyst_i, X - trend_init)}")
    print(f"R2 detrend: {r2(nyst_i, X - trend_hat)}")
    print(f"R2 denoise: {res_trial['r2_full']}")
    print(f"R2 denoise init: {res_trial['r2_init']}")
    print(f"R2 denoise no trend: {res_trial['r2_no']}")
    print(f"Corr D_hat: {res_trial['corr_full']}")
    print(f"Corr D_hat init: {res_trial['corr_init']}")
    print(f"Corr D_hat no trend: {res_trial['corr_no']}")
    print("=" * 80)

    if DEBUG:
        plt.subplot(1, 2, 1)
        plt.plot(X)
        plt.plot(trend_hat)

        plt.subplot(1, 2, 2)
        plt.plot(nyst_i)
        plt.plot(X - trend_hat)

        plt.figure()
        plt.plot(d_hat.T)
        plt.plot(nyst_patterns[i])
        plt.show()

        import IPython; IPython.embed(colors='neutral')
        raise SystemExit(0)

try:
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_pickle('results.pkl')
except Exception:
    import IPython; IPython.embed(colors='neutral')
