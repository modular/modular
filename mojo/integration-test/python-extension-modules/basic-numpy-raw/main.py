# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %bare-mojo build %S/mojo_module.mojo --emit shared-lib -o mojo_module.so
# RUN: python3 %s

import sys

import numpy as np

# Put the current directory (containing module.so) on the Python module lookup
# path.
sys.path.insert(0, "")

# Imports from 'mojo_module.so'
import mojo_module

if __name__ == "__main__":
    print("Hello from Basic Numpy Example!")

    enumerated = np.empty((5, 5), dtype=np.int32)
    for i, j in np.ndindex(enumerated.shape):
        enumerated[i, j] = 10 * i + j

    print(f"The original array has contents: \n{enumerated}")

    expected = np.array(
        [
            [0, 1, 2, 3, 4],
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24],
            [30, 31, 32, 33, 34],
            [40, 41, 42, 43, 44],
        ]
    )
    assert np.array_equal(enumerated, expected)

    mojo_module.mojo_incr_np_array(enumerated)

    print(f"The altered array has contents: \n{enumerated}")

    expected = np.array(
        [
            [1, 2, 3, 4, 5],
            [11, 12, 13, 14, 15],
            [21, 22, 23, 24, 25],
            [31, 32, 33, 34, 35],
            [41, 42, 43, 44, 45],
        ]
    )
    assert np.array_equal(enumerated, expected)

    print("🎉🎉🎉 Mission Success! 🎉🎉🎉")
