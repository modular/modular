# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s

from Buffer import NDBuffer, Buffer, _raw_stack_allocation
from Range import range
from DType import DType
from Functional import elementwise
from Math import mul, min
from List import Dim, DimList, create_dim_list
from IO import print
from Index import StaticIntTuple
from LLCL import Runtime, OwningOutputChainPtr
from TypeUtilities import rebind
from SIMD import F32
from SIMD import SIMD


fn test_elementwise[
    numelems: Int, outer_rank: Int, static_shape: DimList[outer_rank]
](dims: DimList[outer_rank]):
    var memory1 = _raw_stack_allocation[numelems, DType.f32, 1]()
    var buffer1 = NDBuffer[
        outer_rank,
        rebind[DimList[outer_rank]](static_shape),
        DType.f32,
    ](
        memory1.address,
        dims,
        DType.f32,
    )

    var memory2 = _raw_stack_allocation[numelems, DType.f32, 1]()
    var buffer2 = NDBuffer[
        outer_rank,
        rebind[DimList[outer_rank]](static_shape),
        DType.f32,
    ](
        memory2.address,
        dims,
        DType.f32,
    )

    var memory3 = _raw_stack_allocation[numelems, DType.f32, 1]()
    var out_buffer = NDBuffer[
        outer_rank,
        rebind[DimList[outer_rank]](static_shape),
        DType.f32,
    ](
        memory3.address,
        dims,
        DType.f32,
    )

    var x: F32 = 1.0
    for i in range(numelems):
        buffer1.data.offset(i).store(2.0)
        buffer2.data.offset(i).store(SIMD[DType.f32, 1](x.value))
        out_buffer.data.offset(i).store(0.0)
        x += 1.0

    @always_inline
    fn func[simd_width: Int, rank: Int](idx: StaticIntTuple[rank]):
        var index = rebind[StaticIntTuple[outer_rank]](idx)
        var in1 = buffer1.simd_load[simd_width](index)
        var in2 = buffer2.simd_load[simd_width](index)
        out_buffer.simd_store[simd_width](index, mul(in1, in2))

    let runtime = Runtime(4)
    let out_chain = OwningOutputChainPtr(runtime)
    elementwise[outer_rank, 1, 1, func](
        rebind[StaticIntTuple[outer_rank]](out_buffer.dynamic_shape),
        out_chain.borrow(),
    )
    out_chain.wait()
    out_chain.__del__()
    runtime.__del__()

    for i2 in range(min(numelems, 64)):
        print(out_buffer.data.offset(i2).load())


fn main():
    print("Testing 1D:")
    test_elementwise[16, 1, DimList[1].create_unknown()](create_dim_list(16))

    print("Testing 2D:")
    test_elementwise[16, 2, DimList[2].create_unknown()](create_dim_list(4, 4))

    print("Testing 3D:")
    test_elementwise[16, 3, DimList[3].create_unknown()](
        create_dim_list(4, 2, 2)
    )

    print("Testing 4D:")
    test_elementwise[32, 4, DimList[4].create_unknown()](
        create_dim_list(4, 2, 2, 2)
    )

    print("Testing 5D:")
    test_elementwise[32, 5, DimList[5].create_unknown()](
        create_dim_list(4, 2, 1, 2, 2)
    )

    print("Testing large:")
    test_elementwise[131072, 2, DimList[2].create_unknown()](
        create_dim_list(1024, 128)
    )


# CHECK: Testing 1D:
# CHECK-NEXT: 2.000000
# CHECK-NEXT: 4.000000
# CHECK-NEXT: 6.000000
# CHECK-NEXT: 8.000000
# CHECK-NEXT: 10.000000
# CHECK-NEXT: 12.000000
# CHECK-NEXT: 14.000000
# CHECK-NEXT: 16.000000
# CHECK-NEXT: 18.000000
# CHECK-NEXT: 20.000000
# CHECK-NEXT: 22.000000
# CHECK-NEXT: 24.000000
# CHECK-NEXT: 26.000000
# CHECK-NEXT: 28.000000
# CHECK-NEXT: 30.000000
# CHECK-NEXT: 32.000000


# CHECK: Testing 2D:
# CHECK-NEXT: 2.000000
# CHECK-NEXT: 4.000000
# CHECK-NEXT: 6.000000
# CHECK-NEXT: 8.000000
# CHECK-NEXT: 10.000000
# CHECK-NEXT: 12.000000
# CHECK-NEXT: 14.000000
# CHECK-NEXT: 16.000000
# CHECK-NEXT: 18.000000
# CHECK-NEXT: 20.000000
# CHECK-NEXT: 22.000000
# CHECK-NEXT: 24.000000
# CHECK-NEXT: 26.000000
# CHECK-NEXT: 28.000000
# CHECK-NEXT: 30.000000
# CHECK-NEXT: 32.000000


# CHECK: Testing 3D:
# CHECK-NEXT: 2.000000
# CHECK-NEXT: 4.000000
# CHECK-NEXT: 6.000000
# CHECK-NEXT: 8.000000
# CHECK-NEXT: 10.000000
# CHECK-NEXT: 12.000000
# CHECK-NEXT: 14.000000
# CHECK-NEXT: 16.000000
# CHECK-NEXT: 18.000000
# CHECK-NEXT: 20.000000
# CHECK-NEXT: 22.000000
# CHECK-NEXT: 24.000000
# CHECK-NEXT: 26.000000
# CHECK-NEXT: 28.000000
# CHECK-NEXT: 30.000000
# CHECK-NEXT: 32.000000


# CHECK: Testing 4D:
# CHECK-NEXT: 2.000000
# CHECK-NEXT: 4.000000
# CHECK-NEXT: 6.000000
# CHECK-NEXT: 8.000000
# CHECK-NEXT: 10.000000
# CHECK-NEXT: 12.000000
# CHECK-NEXT: 14.000000
# CHECK-NEXT: 16.000000
# CHECK-NEXT: 18.000000
# CHECK-NEXT: 20.000000
# CHECK-NEXT: 22.000000
# CHECK-NEXT: 24.000000
# CHECK-NEXT: 26.000000
# CHECK-NEXT: 28.000000
# CHECK-NEXT: 30.000000
# CHECK-NEXT: 32.000000
# CHECK-NEXT: 34.000000
# CHECK-NEXT: 36.000000
# CHECK-NEXT: 38.000000
# CHECK-NEXT: 40.000000
# CHECK-NEXT: 42.000000
# CHECK-NEXT: 44.000000
# CHECK-NEXT: 46.000000
# CHECK-NEXT: 48.000000
# CHECK-NEXT: 50.000000
# CHECK-NEXT: 52.000000
# CHECK-NEXT: 54.000000
# CHECK-NEXT: 56.000000
# CHECK-NEXT: 58.000000
# CHECK-NEXT: 60.000000
# CHECK-NEXT: 62.000000
# CHECK-NEXT: 64.000000


# CHECK: Testing 5D:
# CHECK-NEXT: 2.000000
# CHECK-NEXT: 4.000000
# CHECK-NEXT: 6.000000
# CHECK-NEXT: 8.000000
# CHECK-NEXT: 10.000000
# CHECK-NEXT: 12.000000
# CHECK-NEXT: 14.000000
# CHECK-NEXT: 16.000000
# CHECK-NEXT: 18.000000
# CHECK-NEXT: 20.000000
# CHECK-NEXT: 22.000000
# CHECK-NEXT: 24.000000
# CHECK-NEXT: 26.000000
# CHECK-NEXT: 28.000000
# CHECK-NEXT: 30.000000
# CHECK-NEXT: 32.000000
# CHECK-NEXT: 34.000000
# CHECK-NEXT: 36.000000
# CHECK-NEXT: 38.000000
# CHECK-NEXT: 40.000000
# CHECK-NEXT: 42.000000
# CHECK-NEXT: 44.000000
# CHECK-NEXT: 46.000000
# CHECK-NEXT: 48.000000
# CHECK-NEXT: 50.000000
# CHECK-NEXT: 52.000000
# CHECK-NEXT: 54.000000
# CHECK-NEXT: 56.000000
# CHECK-NEXT: 58.000000
# CHECK-NEXT: 60.000000
# CHECK-NEXT: 62.000000
# CHECK-NEXT: 64.000000

# CHECK: Testing large:
# CHECK-NEXT: 2.000000
# CHECK-NEXT: 4.000000
# CHECK-NEXT: 6.000000
# CHECK-NEXT: 8.000000
# CHECK-NEXT: 10.000000
# CHECK-NEXT: 12.000000
# CHECK-NEXT: 14.000000
# CHECK-NEXT: 16.000000
# CHECK-NEXT: 18.000000
# CHECK-NEXT: 20.000000
# CHECK-NEXT: 22.000000
# CHECK-NEXT: 24.000000
# CHECK-NEXT: 26.000000
# CHECK-NEXT: 28.000000
# CHECK-NEXT: 30.000000
# CHECK-NEXT: 32.000000
# CHECK-NEXT: 34.000000
# CHECK-NEXT: 36.000000
# CHECK-NEXT: 38.000000
# CHECK-NEXT: 40.000000
# CHECK-NEXT: 42.000000
# CHECK-NEXT: 44.000000
# CHECK-NEXT: 46.000000
# CHECK-NEXT: 48.000000
# CHECK-NEXT: 50.000000
# CHECK-NEXT: 52.000000
# CHECK-NEXT: 54.000000
# CHECK-NEXT: 56.000000
# CHECK-NEXT: 58.000000
# CHECK-NEXT: 60.000000
# CHECK-NEXT: 62.000000
# CHECK-NEXT: 64.000000
# CHECK-NEXT: 66.000000
# CHECK-NEXT: 68.000000
# CHECK-NEXT: 70.000000
# CHECK-NEXT: 72.000000
# CHECK-NEXT: 74.000000
# CHECK-NEXT: 76.000000
# CHECK-NEXT: 78.000000
# CHECK-NEXT: 80.000000
# CHECK-NEXT: 82.000000
# CHECK-NEXT: 84.000000
# CHECK-NEXT: 86.000000
# CHECK-NEXT: 88.000000
# CHECK-NEXT: 90.000000
# CHECK-NEXT: 92.000000
# CHECK-NEXT: 94.000000
# CHECK-NEXT: 96.000000
# CHECK-NEXT: 98.000000
# CHECK-NEXT: 100.000000
# CHECK-NEXT: 102.000000
# CHECK-NEXT: 104.000000
# CHECK-NEXT: 106.000000
# CHECK-NEXT: 108.000000
# CHECK-NEXT: 110.000000
# CHECK-NEXT: 112.000000
# CHECK-NEXT: 114.000000
# CHECK-NEXT: 116.000000
# CHECK-NEXT: 118.000000
# CHECK-NEXT: 120.000000
# CHECK-NEXT: 122.000000
# CHECK-NEXT: 124.000000
# CHECK-NEXT: 126.000000
# CHECK-NEXT: 128.000000
