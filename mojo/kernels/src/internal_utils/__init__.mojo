# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from ._measure import correlation, cosine, kl_div
from ._testing import (
    assert_almost_equal,
    assert_equal,
    assert_with_measure,
    compare,
)
from ._utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    TestTensor,
    bench_compile_time,
    env_get_bool,
    env_get_dtype,
    env_get_shape,
    fill,
    int_list_to_tuple,
    linspace,
    parse_shape,
    random,
    zero,
)
