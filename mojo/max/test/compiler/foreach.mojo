# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo -D MOJO_ENABLE_ASSERTIONS %s

from max.compiler import foreach

from max._driver import cpu_device, UnsafeTensorSlice
from utils import Index
from max.tensor import TensorSpec
from testing import assert_equal


# CHECK-LABEL: == test_foreach
fn test_foreach() raises:
    print("== test_foreach")
    var dev = cpu_device()
    var shape = Index(10, 2)

    var data1 = dev.allocate(TensorSpec(DType.float32, shape))
    var tensor1 = data1^.to_tensor[DType.float32, shape.size]()
    var unsafe_slice1 = UnsafeTensorSlice[DType.float32, 2](
        tensor1.unsafe_ptr(), shape
    )

    var data2 = dev.allocate(TensorSpec(DType.float32, shape))
    var tensor2 = data2^.to_tensor[DType.float32, shape.size]()
    var unsafe_slice2 = UnsafeTensorSlice[DType.float32, 2](
        tensor2.unsafe_ptr(), shape
    )

    # Testing in place modifications.
    @always_inline
    @parameter
    fn set_to_one[
        simd_width: Int
    ](idx: StaticIntTuple[2]) -> SIMD[DType.float32, simd_width]:
        return SIMD[DType.float32, simd_width](1)

    foreach[set_to_one](unsafe_slice1)

    for i in range(10):
        for j in range(2):
            assert_equal(unsafe_slice1[i, j], 1)
            assert_equal(tensor1[i, j], 1)

    # Testing the capture of unsafe_slice1.
    @always_inline
    @parameter
    fn double_it[
        simd_width: Int
    ](idx: StaticIntTuple[2]) -> SIMD[DType.float32, simd_width]:
        return 2 * unsafe_slice1.load[simd_width](idx)

    foreach[double_it](unsafe_slice2)

    for i in range(10):
        for j in range(2):
            assert_equal(unsafe_slice2[i, j], 2)
            assert_equal(tensor2[i, j], 2)

    _ = tensor1^
    _ = tensor2^


fn main() raises:
    test_foreach()
