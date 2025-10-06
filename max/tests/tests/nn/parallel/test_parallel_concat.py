# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import numpy as np
import pytest
from max.nn.parallel import ParallelArrayOps
from numpy.testing import assert_equal


@pytest.mark.parametrize("axis", [-1, 0, 1, 2])
@pytest.mark.parametrize(
    "shape", [(1000, 1000, 3), (1000, 1000), (1000, 1000, 3, 3)]
)
@pytest.mark.parametrize("num_arrays", [2, 3, 4])
def test_parallel_concat(
    axis: int, shape: tuple[int, ...], num_arrays: int
) -> None:
    parallel_ops = ParallelArrayOps()

    arrays = [np.random.rand(*shape) for _ in range(num_arrays)]

    # Validate axis is within bounds
    if axis < -len(shape) or axis >= len(shape):
        with pytest.raises(IndexError):
            out = parallel_ops.concatenate(arrays, axis=axis)
        return

    out = parallel_ops.concatenate(arrays, axis=axis)
    out_np = np.concatenate(arrays, axis=axis)

    assert_equal(out, out_np)
