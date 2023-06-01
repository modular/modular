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
from List import Dim, DimList
from IO import print
from Index import StaticIntTuple
from LLCL import Runtime, OwningOutputChainPtr
from TypeUtilities import rebind
from SIMD import Float32
from SIMD import SIMD


fn test_elementwise[
    numelems: Int, outer_rank: Int, static_shape: DimList
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
        var index = rebind[StaticIntTuple[outer_rank]](idx)
        var in1 = buffer1.simd_load[simd_width](index)
        var in2 = buffer2.simd_load[simd_width](index)
        out_buffer.simd_store[simd_width](index, mul(in1, in2))

    with Runtime(4) as runtime:
        let out_chain = OwningOutputChainPtr(runtime)
        elementwise[outer_rank, 1, 1, func](
            rebind[StaticIntTuple[outer_rank]](out_buffer.dynamic_shape),
            out_chain.borrow(),
        )
        out_chain.wait()

    for i2 in range(min(numelems, 64)):
        print(out_buffer.data.offset(i2).load())


fn main():
    print("Testing 1D:")
    test_elementwise[16, 1, DimList.create_unknown[1]()](DimList(16))

    print("Testing 2D:")
    test_elementwise[16, 2, DimList.create_unknown[2]()](DimList(4, 4))

    print("Testing 3D:")
    test_elementwise[16, 3, DimList.create_unknown[3]()](DimList(4, 2, 2))

    print("Testing 4D:")
    test_elementwise[32, 4, DimList.create_unknown[4]()](DimList(4, 2, 2, 2))

    print("Testing 5D:")
    test_elementwise[32, 5, DimList.create_unknown[5]()](DimList(4, 2, 1, 2, 2))

    print("Testing large:")
    test_elementwise[131072, 2, DimList.create_unknown[2]()](DimList(1024, 128))


# CHECK: Testing 1D:
# CHECK-NEXT: 2.0
# CHECK-NEXT: 4.0
# CHECK-NEXT: 6.0
# CHECK-NEXT: 8.0
# CHECK-NEXT: 10.0
# CHECK-NEXT: 12.0
# CHECK-NEXT: 14.0
# CHECK-NEXT: 16.0
# CHECK-NEXT: 18.0
# CHECK-NEXT: 20.0
# CHECK-NEXT: 22.0
# CHECK-NEXT: 24.0
# CHECK-NEXT: 26.0
# CHECK-NEXT: 28.0
# CHECK-NEXT: 30.0
# CHECK-NEXT: 32.0


# CHECK: Testing 2D:
# CHECK-NEXT: 2.0
# CHECK-NEXT: 4.0
# CHECK-NEXT: 6.0
# CHECK-NEXT: 8.0
# CHECK-NEXT: 10.0
# CHECK-NEXT: 12.0
# CHECK-NEXT: 14.0
# CHECK-NEXT: 16.0
# CHECK-NEXT: 18.0
# CHECK-NEXT: 20.0
# CHECK-NEXT: 22.0
# CHECK-NEXT: 24.0
# CHECK-NEXT: 26.0
# CHECK-NEXT: 28.0
# CHECK-NEXT: 30.0
# CHECK-NEXT: 32.0


# CHECK: Testing 3D:
# CHECK-NEXT: 2.0
# CHECK-NEXT: 4.0
# CHECK-NEXT: 6.0
# CHECK-NEXT: 8.0
# CHECK-NEXT: 10.
# CHECK-NEXT: 12.0
# CHECK-NEXT: 14.0
# CHECK-NEXT: 16.0
# CHECK-NEXT: 18.0
# CHECK-NEXT: 20.0
# CHECK-NEXT: 22.0
# CHECK-NEXT: 24.0
# CHECK-NEXT: 26.0
# CHECK-NEXT: 28.0
# CHECK-NEXT: 30.0
# CHECK-NEXT: 32.0


# CHECK: Testing 4D:
# CHECK-NEXT: 2.0
# CHECK-NEXT: 4.0
# CHECK-NEXT: 6.0
# CHECK-NEXT: 8.0
# CHECK-NEXT: 10.0
# CHECK-NEXT: 12.0
# CHECK-NEXT: 14.0
# CHECK-NEXT: 16.0
# CHECK-NEXT: 18.0
# CHECK-NEXT: 20.0
# CHECK-NEXT: 22.0
# CHECK-NEXT: 24.0
# CHECK-NEXT: 26.0
# CHECK-NEXT: 28.0
# CHECK-NEXT: 30.0
# CHECK-NEXT: 32.0
# CHECK-NEXT: 34.0
# CHECK-NEXT: 36.0
# CHECK-NEXT: 38.0
# CHECK-NEXT: 40.0
# CHECK-NEXT: 42.0
# CHECK-NEXT: 44.0
# CHECK-NEXT: 46.0
# CHECK-NEXT: 48.0
# CHECK-NEXT: 50.0
# CHECK-NEXT: 52.0
# CHECK-NEXT: 54.0
# CHECK-NEXT: 56.0
# CHECK-NEXT: 58.0
# CHECK-NEXT: 60.0
# CHECK-NEXT: 62.0
# CHECK-NEXT: 64.0


# CHECK: Testing 5D:
# CHECK-NEXT: 2.0
# CHECK-NEXT: 4.0
# CHECK-NEXT: 6.0
# CHECK-NEXT: 8.0
# CHECK-NEXT: 10.0
# CHECK-NEXT: 12.0
# CHECK-NEXT: 14.0
# CHECK-NEXT: 16.0
# CHECK-NEXT: 18.0
# CHECK-NEXT: 20.0
# CHECK-NEXT: 22.0
# CHECK-NEXT: 24.0
# CHECK-NEXT: 26.0
# CHECK-NEXT: 28.0
# CHECK-NEXT: 30.0
# CHECK-NEXT: 32.0
# CHECK-NEXT: 34.0
# CHECK-NEXT: 36.0
# CHECK-NEXT: 38.0
# CHECK-NEXT: 40.0
# CHECK-NEXT: 42.0
# CHECK-NEXT: 44.0
# CHECK-NEXT: 46.0
# CHECK-NEXT: 48.0
# CHECK-NEXT: 50.0
# CHECK-NEXT: 52.0
# CHECK-NEXT: 54.0
# CHECK-NEXT: 56.0
# CHECK-NEXT: 58.0
# CHECK-NEXT: 60.0
# CHECK-NEXT: 62.0
# CHECK-NEXT: 64.0

# CHECK: Testing large:
# CHECK-NEXT: 2.0
# CHECK-NEXT: 4.0
# CHECK-NEXT: 6.0
# CHECK-NEXT: 8.0
# CHECK-NEXT: 10.0
# CHECK-NEXT: 12.0
# CHECK-NEXT: 14.0
# CHECK-NEXT: 16.0
# CHECK-NEXT: 18.0
# CHECK-NEXT: 20.0
# CHECK-NEXT: 22.0
# CHECK-NEXT: 24.0
# CHECK-NEXT: 26.0
# CHECK-NEXT: 28.0
# CHECK-NEXT: 30.0
# CHECK-NEXT: 32.0
# CHECK-NEXT: 34.0
# CHECK-NEXT: 36.0
# CHECK-NEXT: 38.0
# CHECK-NEXT: 40.0
# CHECK-NEXT: 42.0
# CHECK-NEXT: 44.0
# CHECK-NEXT: 46.0
# CHECK-NEXT: 48.0
# CHECK-NEXT: 50.0
# CHECK-NEXT: 52.0
# CHECK-NEXT: 54.0
# CHECK-NEXT: 56.0
# CHECK-NEXT: 58.0
# CHECK-NEXT: 60.0
# CHECK-NEXT: 62.0
# CHECK-NEXT: 64.0
# CHECK-NEXT: 66.0
# CHECK-NEXT: 68.0
# CHECK-NEXT: 70.0
# CHECK-NEXT: 72.0
# CHECK-NEXT: 74.0
# CHECK-NEXT: 76.0
# CHECK-NEXT: 78.0
# CHECK-NEXT: 80.0
# CHECK-NEXT: 82.0
# CHECK-NEXT: 84.0
# CHECK-NEXT: 86.0
# CHECK-NEXT: 88.0
# CHECK-NEXT: 90.0
# CHECK-NEXT: 92.0
# CHECK-NEXT: 94.0
# CHECK-NEXT: 96.0
# CHECK-NEXT: 98.0
# CHECK-NEXT: 100.0
# CHECK-NEXT: 102.0
# CHECK-NEXT: 104.0
# CHECK-NEXT: 106.0
# CHECK-NEXT: 108.0
# CHECK-NEXT: 110.0
# CHECK-NEXT: 112.0
# CHECK-NEXT: 114.0
# CHECK-NEXT: 116.0
# CHECK-NEXT: 118.0
# CHECK-NEXT: 120.0
# CHECK-NEXT: 122.0
# CHECK-NEXT: 124.0
# CHECK-NEXT: 126.0
# CHECK-NEXT: 128.0
