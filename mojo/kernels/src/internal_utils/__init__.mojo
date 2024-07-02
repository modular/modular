# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from ._testing import assert_almost_equal, assert_equal, compare
from ._utils import (
    DeviceNDBuffer,
    fill,
    HostNDBuffer,
    linspace,
    TestTensor,
    zero,
    random,
)
from ._measure import kl_div, correlation, cosine
