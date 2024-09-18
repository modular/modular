# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: system-windows
# UNSUPPORTED: asan
# RUN: %mojo -debug-level full %s

from max.engine import EngineNumpyView
from python import Python
from testing import assert_equal


fn test_numpy_view() raises:
    var np = Python.import_module("numpy")

    var n1 = np.array([1, 2, 3]).astype(np.float32)

    var n1_view = EngineNumpyView(n1)

    assert_equal(str(n1_view.spec()), "3xfloat32")

    _ = n1^


fn main() raises:
    test_numpy_view()
