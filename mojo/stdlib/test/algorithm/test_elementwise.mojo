# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s
# RUN: %mojo %s -debug-level=full | FileCheck %s

from Buffer import NDBuffer, Buffer, _raw_stack_allocation
from Range import range
from DType import DType
from Functional import _elementwise_impl, _get_start_indices_of_nth_subvolume
from Math import mul, min
from List import Dim, DimList
from IO import print
from Index import StaticIntTuple
from LLCL import Runtime, OutputChainPtr, OwningOutputChainPtr
from TypeUtilities import rebind
from SIMD import Float32
from SIMD import SIMD


fn test_elementwise[
    numelems: Int, outer_rank: Int, static_shape: DimList, is_blocking: Bool
](dims: DimList):
    var memory1 = _raw_stack_allocation[numelems, DType.float32, 1]()
    var buffer1 = NDBuffer[
        outer_rank,
        rebind[DimList](static_shape),
        DType.float32,
    ](
        memory1.address,
        dims,
        DType.float32,
    )

    var memory2 = _raw_stack_allocation[numelems, DType.float32, 1]()
    var buffer2 = NDBuffer[
        outer_rank,
        rebind[DimList](static_shape),
        DType.float32,
    ](
        memory2.address,
        dims,
        DType.float32,
    )

    var memory3 = _raw_stack_allocation[numelems, DType.float32, 1]()
    var out_buffer = NDBuffer[
        outer_rank,
        rebind[DimList](static_shape),
        DType.float32,
    ](
        memory3.address,
        dims,
        DType.float32,
    )

    var x: Float32 = 1.0
    for i in range(numelems):
        buffer1.data.offset(i).store(2.0)
        buffer2.data.offset(i).store(SIMD[DType.float32, 1](x.value))
        out_buffer.data.offset(i).store(0.0)
        x += 1.0

    @always_inline
    @parameter
    fn func[simd_width: Int, rank: Int](idx: StaticIntTuple[rank]):
        let index = rebind[StaticIntTuple[outer_rank]](idx)
        let in1 = buffer1.simd_load[simd_width](index)
        let in2 = buffer2.simd_load[simd_width](index)
        out_buffer.simd_store[simd_width](index, mul(in1, in2))

    with Runtime(4) as runtime:

        @parameter
        if is_blocking:
            _elementwise_impl[outer_rank, 1, is_blocking, func](
                rebind[StaticIntTuple[outer_rank]](out_buffer.dynamic_shape),
                OutputChainPtr(),
            )
        else:
            let out_chain = OwningOutputChainPtr(runtime)
            _elementwise_impl[outer_rank, 1, is_blocking, func](
                rebind[StaticIntTuple[outer_rank]](out_buffer.dynamic_shape),
                out_chain.borrow(),
            )
            out_chain.wait()

    for i2 in range(min(numelems, 64)):
        if out_buffer.data.offset(i2).load() != 2 * (i2 + 1):
            print("ERROR")


fn test_indices_conversion():
    print("== Testing indices conversion:")
    let shape = StaticIntTuple[4](3, 4, 5, 6)
    print(_get_start_indices_of_nth_subvolume[4, 0](10, shape))
    print(_get_start_indices_of_nth_subvolume[4, 1](10, shape))
    print(_get_start_indices_of_nth_subvolume[4, 2](10, shape))
    print(_get_start_indices_of_nth_subvolume[4, 3](2, shape))
    print(_get_start_indices_of_nth_subvolume[4, 4](0, shape))


fn main():
    # CHECK-LABEL: == Testing 1D:
    # CHECK-NOT: ERROR
    print("== Testing 1D:")
    test_elementwise[16, 1, DimList.create_unknown[1](), False](DimList(16))

    # CHECK-LABEL: == Testing 1D blocking:
    # CHECK-NOT: ERROR
    print("== Testing 1D blocking:")
    test_elementwise[16, 1, DimList.create_unknown[1](), True](DimList(16))

    # CHECK-LABEL: == Testing 2D:
    # CHECK-NOT: ERROR
    print("== Testing 2D:")
    test_elementwise[16, 2, DimList.create_unknown[2](), False](DimList(4, 4))

    # CHECK-LABEL: == Testing 2D blocking:
    # CHECK-NOT: ERROR
    print("== Testing 2D blocking:")
    test_elementwise[16, 2, DimList.create_unknown[2](), True](DimList(4, 4))

    # CHECK-LABEL: == Testing 3D:
    # CHECK-NOT: ERROR
    print("== Testing 3D:")
    test_elementwise[16, 3, DimList.create_unknown[3](), False](
        DimList(4, 2, 2)
    )

    # CHECK-LABEL: == Testing 3D blocking:
    # CHECK-NOT: ERROR
    print("== Testing 3D blocking:")
    test_elementwise[16, 3, DimList.create_unknown[3](), True](DimList(4, 2, 2))

    # CHECK-LABEL: == Testing 4D:
    # CHECK-NOT: ERROR
    print("== Testing 4D:")
    test_elementwise[32, 4, DimList.create_unknown[4](), False](
        DimList(4, 2, 2, 2)
    )

    # CHECK-LABEL: == Testing 4D blocking:
    # CHECK-NOT: ERROR
    print("== Testing 4D blocking:")
    test_elementwise[32, 4, DimList.create_unknown[4](), False](
        DimList(4, 2, 2, 2)
    )

    # CHECK-LABEL: == Testing 5D:
    # CHECK-NOT: ERROR
    print("== Testing 5D:")
    test_elementwise[32, 5, DimList.create_unknown[5](), False](
        DimList(4, 2, 1, 2, 2)
    )

    # CHECK-LABEL: == Testing 5D blocking:
    # CHECK-NOT: ERROR
    print("== Testing 5D blocking:")
    test_elementwise[32, 5, DimList.create_unknown[5](), True](
        DimList(4, 2, 1, 2, 2)
    )

    # CHECK-LABEL: == Testing large:
    # CHECK-NOT: ERROR
    print("== Testing large:")
    test_elementwise[131072, 2, DimList.create_unknown[2](), False](
        DimList(1024, 128)
    )

    # CHECK-LABEL: == Testing large blocking:
    # CHECK-NOT: ERROR
    print("== Testing large blocking:")
    test_elementwise[131072, 2, DimList.create_unknown[2](), True](
        DimList(1024, 128)
    )

    # CHECK-LABEL: == Testing indices conversion:
    # CHECK: (0, 0, 1, 4)
    # CHECK: (0, 2, 0, 0)
    # CHECK: (2, 2, 0, 0)
    # CHECK: (2, 0, 0, 0)
    # CHECK: (0, 0, 0, 0)
    test_indices_conversion()
