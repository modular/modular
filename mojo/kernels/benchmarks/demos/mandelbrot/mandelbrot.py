# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import time

MAX = 1000  # maximum Mandelbrot iterations
N = 4096  # width and height of image
pixels = [0 for _ in range(N * N)]


def scale(x, a, b):
    return a + (x / N) * (b - a)


t0 = time.time()
for i in range(N):
    for j in range(N):
        c = complex(scale(j, -2.00, 0.47), scale(i, -1.12, 1.12))
        z = 0j
        iteration = 0

        while abs(z) <= 2 and iteration < MAX:
            z = z**2 + c
            iteration += 1

        # pixels[i*N + j] = int(255 * iteration/MAX)
        pixels[i * N + j] = iteration
print(sum(pixels))
print(time.time() - t0)
