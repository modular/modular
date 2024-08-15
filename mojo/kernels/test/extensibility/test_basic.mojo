# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from extensibility import Tensor, empty_tensor

from utils import StaticIntTuple


fn ones[
    type: DType, rank: Int
](shape: StaticIntTuple[rank]) -> Tensor[type, rank]:
    var output = empty_tensor[type](shape)

    @always_inline
    @parameter
    fn func[width: Int](i: StaticIntTuple[rank]) -> SIMD[type, width]:
        return 1

    output.for_each[func]()
    return output^


fn my_add[
    type: DType, rank: Int
](x: Tensor[type, rank], y: Tensor[type, rank]) -> Tensor[type, rank]:
    var out = empty_tensor[type, rank](x.shape)

    @parameter
    @always_inline
    fn func[width: Int](i: StaticIntTuple[rank]) -> SIMD[type, width]:
        return x.simd_load[width](i) + y.simd_load[width](i)

    out.for_each[func]()
    return out^


# CHECK-LABEL: == test_add
fn test_add():
    print("== test_add")

    var shape = StaticIntTuple[2](5, 2)
    var t1 = ones[DType.float32, 2](shape)
    var t2 = ones[DType.float32, 2](shape)

    var t3 = my_add(t1, t2)
    var t4 = my_add(t3, t3)

    for i in range(t4.nelems()):
        # Convert to flat indices.
        var indices = StaticIntTuple[2](
            (i // t4.shape[1]) % t4.shape[0], i % t4.shape[1]
        )
        print(t4.simd_load[1](indices))

    # CHECK-NEXT: 4.0
    # CHECK-NEXT: 4.0
    # CHECK-NEXT: 4.0
    # CHECK-NEXT: 4.0
    # CHECK-NEXT: 4.0
    # CHECK-NEXT: 4.0
    # CHECK-NEXT: 4.0
    # CHECK-NEXT: 4.0
    # CHECK-NEXT: 4.0
    # CHECK-NEXT: 4.0


# CHECK-LABEL: == test_default_strides
fn test_default_strides():
    print("== test_default_strides")
    var shape = StaticIntTuple[3](5, 3, 2)
    var output = empty_tensor[DType.float32](shape)

    # CHECK-NEXT: 3
    # CHECK-NEXT: 3
    print(len(output.shape))
    print(len(output.strides))

    # CHECK-NEXT: (6, 2, 1)
    print(output.strides)


# CHECK-LABEL: == test_scalar_index_access
fn test_scalar_index_access():
    print("== test_scalar_index_access")

    var shape = StaticIntTuple[1](4)
    var tensor = empty_tensor[DType.int32](shape)

    tensor.store(0, Int32(42))
    tensor.store(1, Int32(123))
    tensor.store(2, Int32(-1344))
    tensor.store(3, Int32(0))
    print(tensor.simd_load[4](0))
    # CHECK-NEXT:[42, 123, -1344, 0]


# CHECK-LABEL: == test_get_nd_indices
fn test_get_nd_indices():
    print("== test_get_nd_indices")
    var shape = StaticIntTuple[7](5, 3, 2, 1, 1, 1, 5)
    var tensor = empty_tensor[DType.int32](shape)
    var indices = tensor.get_nd_indices()
    print(indices)
    # CHECK-NEXT:(0, 0, 0, 0, 0, 0, 0)


fn main():
    test_add()
    test_default_strides()
    test_scalar_index_access()
    test_get_nd_indices()
