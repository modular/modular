# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test eager tensors basic behavior."""

import numpy as np
import pytest
from hypothesis import Phase, given, settings
from hypothesis.extra import numpy as nst
from max.dtype import DType
from max.experimental.eager import Tensor


@given(
    na=nst.arrays(dtype=np.float32, shape=(3, 1)),
    nb=nst.arrays(dtype=np.float32, shape=(3, 1)),
    nc=nst.arrays(dtype=np.float32, shape=(1, 3)),
    nd=nst.arrays(dtype=np.float32, shape=(1, 3)),
)
# currently slooooow
@settings(max_examples=5, phases=[Phase.generate], deadline=None)
@pytest.mark.asyncio
async def test_basic_add(na, nb, nc, nd):
    a, b, c, d = (Tensor.from_numpy(n) for n in (na, nb, nc, nd))
    expected = (na + nb) + (nc + nd)
    value = (a + b) + (c + d)
    assert value.shape == [3, 3]
    assert value.dtype == DType.float32
    await value
    result = value.to_numpy()
    np.testing.assert_allclose(result, expected)


@given(
    na=nst.arrays(dtype=np.float32, shape=(3, 2)),
    nb=nst.arrays(dtype=np.float32, shape=(2, 3)),
)
def test_eager_shape_failure(na, nb):
    a, b = Tensor.from_numpy(na), Tensor.from_numpy(nb)
    with pytest.raises(ValueError):
        _ = a + b
