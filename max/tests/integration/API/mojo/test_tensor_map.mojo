# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: %mojo -I %engine_pkg_dir %s | FileCheck %s

from aiengine import InferenceSession
from tensor import TensorSpec, Tensor, TensorShape
from utils.index import Index


fn test_tensor_map() raises:
    # CHECK: test_tensor_map
    print("====test_tensor_map")

    var t1 = Tensor[DType.float32](TensorShape(2, 3))

    for i in range(2):
        for j in range(3):
            t1[Index(i, j)] = 1

    let session = InferenceSession()
    let map = session.new_tensor_map()

    # CHECK: 0
    print(map.__len__())

    map.borrow("tensor", t1)

    # CHECK: 1
    print(map.__len__())
    let t2 = map.get[DType.float32]("tensor")

    # CHECK: 1
    print(map.__len__())

    let t3 = map.get[DType.float32]("tensor")

    # CHECK: True
    print(t1 == t2)

    # CHECK: True
    print(t1 == t3)


fn main() raises:
    test_tensor_map()
