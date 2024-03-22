from time import sleep, time

import numpy as np
import pandas as pd

from mlxvsnumba.mlx_simul import compute_tstats_white as tstats_white_mlx
from mlxvsnumba.mlx_simul import simulate_ar3_garch11 as simul_ar3g_mlx
from mlxvsnumba.numba_simul import compute_tstats_white as tstats_white_numba
from mlxvsnumba.numba_simul import simulate_ar3_garch11 as simul_ar3g_numba
from mlxvsnumba.numpy_simul import compute_tstats_white as tstats_white_numpy
from mlxvsnumba.numpy_simul import simulate_ar3_garch11 as simul_ar3g_numpy

times = []

T_range = [250, 500]

N_range = [
    1000,
    10_000,
    100_000,
    1_000_000,
]

model = {
    "AR3-GARCH": {
        "phi1": 0.5,
        "phi2": 0.2,
        "phi3": 0.3,
        "omega": 0.1,
        "alpha": 0.1,
        "beta": 0.8,
    }
}

# Pre-compile the numba JIT functions
x = simul_ar3g_numba(1, 250, **model["AR3-GARCH"])
x = simul_ar3g_numba(1, 500, **model["AR3-GARCH"])
y = tstats_white_numba(x)

sleep(2)


for N in N_range:
    for T in T_range:
        print(f"\n\n###\n\nT = {T}, N = {N:,}")

        start = time()
        x = simul_ar3g_numpy(N, T, **model["AR3-GARCH"])
        y = tstats_white_numpy(x)
        times.append(
            {
                "t": T,
                "n": N,
                "library": "numpy",
                "time": (exec_time := time() - start),
            }
        )
        del x, y

        print(f"Elapsed (numpy) = {exec_time:0.2f} seconds")

        start = time()
        x = simul_ar3g_numba(N, T, **model["AR3-GARCH"])
        y = tstats_white_numba(x)
        times.append(
            {
                "t": T,
                "n": N,
                "library": "numba",
                "time": (exec_time := time() - start),
            }
        )
        del x, y

        print(f"Elapsed (numba) = {exec_time:0.2f} seconds")

        start = time()
        x = simul_ar3g_mlx(N, T, **model["AR3-GARCH"])
        y = tstats_white_mlx(x)
        z = np.array(y)
        times.append(
            {
                "t": T,
                "n": N,
                "library": "mlx",
                "time": (exec_time := time() - start),
            }
        )
        del x, y

        print(f"Elapsed (mlx) = {exec_time:0.2f} seconds")


df = pd.DataFrame(times)
df.to_csv("times.csv", index=False)
