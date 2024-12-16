# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import time

import numpy as np

n = 4096
m = 4096
MAX_ITER = 1000


def scale(x, a, b):
    return a + (x / n) * (b - a)


cs = np.zeros((n, m), dtype=complex)
zs = np.zeros((n, m), dtype=complex)
for i in range(n):
    for j in range(m):
        cs[i, j] = complex(scale(j, -2.00, 0.47), scale(i, -1.12, 1.12))

mask = np.full((n, m), True, dtype=bool)

t0 = time.time()

for i in range(MAX_ITER):
    zs[mask] = zs[mask] * zs[mask] + cs[mask]
    mask[np.abs(zs) > 2] = False

print(time.time() - t0)
