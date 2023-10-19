# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: %mojo -I %engine_pkg_dir -I %test_utils_pkg_dir %s | FileCheck %s

from aiengine import EngineTensorView
from tensor import Tensor
from test_utils import linear_fill
from python import Python


fn test_tensor_view() raises:
    # CHECK: test_tensor_view
    print("====test_tensor_view")
    var t1 = Tensor[DType.float32](3)
    linear_fill(t1, 1.0, 2.0, 3.0)

    let t1_view = EngineTensorView(t1)

    # CHECK: True
    print(t1.data() == t1_view.data[DType.float32]())

    # CHECK: True
    print(t1.spec() == t1_view.spec())


fn main() raises:
    test_tensor_view()
