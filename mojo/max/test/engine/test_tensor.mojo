# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: %mojo -I %engine_pkg_dir -I %test_utils_pkg_dir %s | FileCheck %s

from max.engine import EngineTensorView, InferenceSession
from tensor import Tensor, TensorShape
from closed_source_test_utils import linear_fill
from utils.index import Index
from python import Python


fn test_tensor_view() raises:
    # CHECK-LABEL: ====test_tensor_view
    print("====test_tensor_view")
    var t1 = Tensor[DType.float32](3)
    linear_fill(t1, 1.0, 2.0, 3.0)

    var t1_view = EngineTensorView(t1)

    # CHECK: True
    print(t1.data() == t1_view.data[DType.float32]())

    # CHECK: True
    print(t1.spec() == t1_view.spec())


fn test_tensor_value() raises:
    # CHECK-LABEL: ====test_tensor_value
    print("====test_tensor_value")

    var t1 = Tensor[DType.float32](TensorShape(2, 3))

    for i in range(2):
        for j in range(3):
            t1[Index(i, j)] = 1

    var session = InferenceSession()
    var value = session.new_borrowed_tensor_value(t1)

    var t2 = value.as_tensor_copy[DType.float32]()

    # CHECK: True
    print(t1 == t2)


fn main() raises:
    test_tensor_view()
    test_tensor_value()
