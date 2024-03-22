import numpy as np
from numba import njit, prange

BURNIN = 100


@njit
def simulate_shocks(N: int, T: int, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    return np.random.normal(mu, sigma, size=(N, T))


@njit
def garch_shocks(
    omega: float, alpha: float, beta: float, shocks: np.ndarray
) -> np.ndarray:
    N, T = shocks.shape
    sigma2 = np.zeros((N, T))
    epsilon = np.zeros((N, T))
    if (alpha + beta) == 1:
        sigma2[:, 0] = omega
    else:
        sigma2[:, 0] = omega / (1 - alpha - beta)

    for t in range(1, T):
        sigma2[:, t] = omega + alpha * epsilon[:, t - 1] ** 2 + beta * sigma2[:, t - 1]
        epsilon[:, t] = np.sqrt(sigma2[:, t]) * shocks[:, t]
    return epsilon


@njit
def ar(phi1: float, phi2: float, phi3: float, shocks: np.ndarray) -> np.ndarray:
    N, T = shocks.shape
    y = np.zeros((N, T))
    p = 3
    y[:, :p] = shocks[:, :p]
    for t in range(p, T):
        y[:, t] = (
            y[:, t - 1] * phi1 + y[:, t - 2] * phi2 + y[:, t - 3] * phi3 + shocks[:, t]
        )
    return y


def simulate_ar3_garch11(
    N: int,
    T: int,
    phi1: float,
    phi2: float,
    phi3: float,
    omega: float,
    alpha: float,
    beta: float,
    burnin: int = BURNIN,
) -> np.ndarray:
    return ar(
        phi1=phi1,
        phi2=phi2,
        phi3=phi3,
        shocks=garch_shocks(
            omega=omega,
            alpha=alpha,
            beta=beta,
            shocks=simulate_shocks(N, T + BURNIN, 0.0, 1.0),
        ),
    )[:, burnin:]


@njit(parallel=True)
def compute_means(x: np.ndarray) -> np.ndarray:
    n, m = x.shape
    means = np.empty(n)
    for i in prange(n):
        means[i] = np.mean(x[i, :])
    return means


@njit(parallel=True)
def compute_std_devs_demeaned(x: np.ndarray) -> np.ndarray:
    n, m = x.shape
    std_devs = np.empty(n)
    for i in prange(n):
        std_devs[i] = np.sqrt(np.sum(x[i, :] ** 2))
    return std_devs


@njit(parallel=True)
def compute_tstats_white(x: np.ndarray) -> np.ndarray:
    n, m = x.shape
    means = compute_means(x)
    std_devs = compute_std_devs_demeaned(x - means.reshape((n, 1)))
    return means / std_devs * m


if __name__ == "__main__":
    import time

    N = 100000
    T = 500
    models = {
        "AR3-GARCH": {
            "phi1": 0.5,
            "phi2": 0.2,
            "phi3": 0.3,
            "omega": 0.1,
            "alpha": 0.1,
            "beta": 0.8,
        }
    }

    start = time.time()
    x = simulate_ar3_garch11(1, T, **models["AR3-GARCH"])
    y = compute_tstats_white(x)
    print(f"Elapsed numba (compilation) = {time.time() - start}")

    start = time.time()
    x = simulate_ar3_garch11(N, T, **models["AR3-GARCH"])
    y = compute_tstats_white(x)
    simul_time = time.time() - start
    print(f"Elapsed numba (after compilation) = {simul_time}")
