import mlx.core as mx
import numpy as np
import statsmodels.api as sm

from .mlx_simul import compute_tstats_white as tstats_white_mlx
from .numba_simul import compute_tstats_white as tstats_white_numba
from .numpy_simul import compute_tstats_white as tstats_white_numpy

"""
Uses statsmodels as the basis for testing the estimators.

"""


def test_white_numpy():
    N = 1000
    T = 500

    np.random.seed(42)
    shocks = np.random.normal(0, 1, size=(N, T))

    tstats = tstats_white_numpy(shocks)

    tstats_sm = np.empty(N)

    for i in range(N):
        reg_res = sm.OLS(shocks[i], np.ones(T)).fit(cov_type="HC0", use_t=True)
        tstats_sm[i] = reg_res.tvalues[0]

    assert np.allclose(tstats, tstats_sm)


def test_white_numba():
    N = 1000
    T = 500

    np.random.seed(42)
    shocks = np.random.normal(0, 1, size=(N, T))

    tstats = tstats_white_numba(shocks)

    tstats_sm = np.empty(N)

    for i in range(N):
        reg_res = sm.OLS(shocks[i], np.ones(T)).fit(cov_type="HC0", use_t=True)
        tstats_sm[i] = reg_res.tvalues[0]

    assert np.allclose(tstats, tstats_sm)


def test_white_mlx():
    N = 1000
    T = 500

    np.random.seed(42)
    shocks = np.random.normal(0, 1, size=(N, T)).astype(np.float32)

    tstats = np.array(tstats_white_mlx(mx.array(shocks.astype(np.float32))))

    tstats_sm = np.empty(N)

    for i in range(N):
        reg_res = sm.OLS(shocks[i], np.ones(T)).fit(cov_type="HC0", use_t=True)
        tstats_sm[i] = reg_res.tvalues[0]

    assert np.allclose(tstats, tstats_sm, atol=1e-4)
