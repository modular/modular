# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: %mojo -debug-level full %s

from testing import assert_equal
from utils.index import Index

from max.engine import InferenceSession
from max.tensor import Tensor, TensorShape


fn test_tensor_map() raises:
    var t1 = Tensor[DType.float32](TensorShape(2, 3))

    for i in range(2):
        for j in range(3):
            t1[Index(i, j)] = 1

    var session = InferenceSession()
    var map = session.new_tensor_map()

    assert_equal(len(map), 0)

    map.borrow("tensor", t1)
    map.borrow(
        "tensor2",
        t1.spec(),
        t1.unsafe_ptr(),
    )

    assert_equal(len(map), 2)
    var t2 = map.get[DType.float32]("tensor")

    assert_equal(len(map), 2)

    var t3 = map.get[DType.float32]("tensor")

    assert_equal(t1, t2)
    assert_equal(t1, t3)

    var t4 = map.get[DType.float32]("tensor2")

    assert_equal(t1, t4)


fn test_tensor_map_value() raises:
    var t1 = Tensor[DType.float32](TensorShape(2, 3))

    for i in range(2):
        for j in range(3):
            t1[Index(i, j)] = 1

    var session = InferenceSession()
    var map = session.new_tensor_map()
    var t1_value = session.new_borrowed_tensor_value(t1)

    assert_equal(len(map), 0)

    map.borrow("tensor", t1_value)

    assert_equal(len(map), 1)

    var t2 = map.get[DType.float32]("tensor")
    assert_equal(t1, t2)

    var t3_value = map.get_value("tensor")
    var t3 = t3_value.as_tensor_copy[DType.float32]()
    assert_equal(t1, t3)

    _ = map^
    _ = t1_value^
    _ = t1^


fn main() raises:
    test_tensor_map()
    test_tensor_map_value()
