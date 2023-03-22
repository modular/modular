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
from Int import Int
from Math import mul
from List import Dim, DimList, create_dim_list
from IO import print
from Index import StaticIntTuple
from LLCL import Runtime
from TypeUtilities import rebind
from F32 import F32
from SIMD import SIMD


fn test_elementwise[
    numelems: Int, outer_rank: Int, static_shape: DimList[outer_rank]
](dims: DimList[outer_rank]):
    var memory1 = _raw_stack_allocation[numelems, DType.f32, 1]()
    var buffer1 = NDBuffer[
        outer_rank.__as_mlir_index(),
        rebind[DimList[outer_rank.__as_mlir_index()]](static_shape),
        DType.f32,
    ](
        memory1.address,
        dims,
        DType.f32,
    )

    var memory2 = _raw_stack_allocation[numelems, DType.f32, 1]()
    var buffer2 = NDBuffer[
        outer_rank.__as_mlir_index(),
        rebind[DimList[outer_rank.__as_mlir_index()]](static_shape),
        DType.f32,
    ](
        memory2.address,
        dims,
        DType.f32,
    )

    var memory3 = _raw_stack_allocation[numelems, DType.f32, 1]()
    var out_buffer = NDBuffer[
        outer_rank.__as_mlir_index(),
        rebind[DimList[outer_rank.__as_mlir_index()]](static_shape),
        DType.f32,
    ](
        memory3.address,
        dims,
        DType.f32,
    )

    var x: F32 = 1.0
    for i in range(numelems):
        buffer1.data.offset(i).store(2.0)
        buffer2.data.offset(i).store(SIMD[1, DType.f32](x.value))
        out_buffer.data.offset(i).store(0.0)
        x += 1.0

    let runtime = Runtime(1)

    @always_inline
    fn func[
        simd_width: Int, rank: __mlir_type.index
    ](idx: StaticIntTuple[rank]):
        var index = rebind[
            StaticIntTuple[Int(outer_rank.__as_mlir_index()).__as_mlir_index()]
        ](idx)
        var in1 = buffer1.simd_load[simd_width](index)
        var in2 = buffer2.simd_load[simd_width](index)
        out_buffer.simd_store[simd_width](index, mul(in1, in2))

    elementwise[outer_rank.__as_mlir_index(), 1, 1, func](
        rebind[StaticIntTuple[outer_rank.__as_mlir_index()]](
            out_buffer.dynamic_shape
        ),
        runtime,
    )

    for i2 in range(numelems):
        print(out_buffer.data.offset(i2).load())


fn main():
    print("Testing 1D:\n")
    test_elementwise[16, 1, DimList[1].create_unknown()](create_dim_list(16))

    print("Testing 2D:\n")
    test_elementwise[16, 2, DimList[2].create_unknown()](create_dim_list(4, 4))

    print("Testing 3D:\n")
    test_elementwise[16, 3, DimList[3].create_unknown()](
        create_dim_list(4, 2, 2)
    )

    print("Testing 4D:\n")
    test_elementwise[32, 4, DimList[4].create_unknown()](
        create_dim_list(4, 2, 2, 2)
    )

    print("Testing 5D:\n")
    test_elementwise[32, 5, DimList[5].create_unknown()](
        create_dim_list(4, 2, 1, 2, 2)
    )


# CHECK: Testing 1D:
# CHECK-NEXT: [2.000000]
# CHECK-NEXT: [4.000000]
# CHECK-NEXT: [6.000000]
# CHECK-NEXT: [8.000000]
# CHECK-NEXT: [10.000000]
# CHECK-NEXT: [12.000000]
# CHECK-NEXT: [14.000000]
# CHECK-NEXT: [16.000000]
# CHECK-NEXT: [18.000000]
# CHECK-NEXT: [20.000000]
# CHECK-NEXT: [22.000000]
# CHECK-NEXT: [24.000000]
# CHECK-NEXT: [26.000000]
# CHECK-NEXT: [28.000000]
# CHECK-NEXT: [30.000000]
# CHECK-NEXT: [32.000000]


# CHECK: Testing 2D:
# CHECK-NEXT: [2.000000]
# CHECK-NEXT: [4.000000]
# CHECK-NEXT: [6.000000]
# CHECK-NEXT: [8.000000]
# CHECK-NEXT: [10.000000]
# CHECK-NEXT: [12.000000]
# CHECK-NEXT: [14.000000]
# CHECK-NEXT: [16.000000]
# CHECK-NEXT: [18.000000]
# CHECK-NEXT: [20.000000]
# CHECK-NEXT: [22.000000]
# CHECK-NEXT: [24.000000]
# CHECK-NEXT: [26.000000]
# CHECK-NEXT: [28.000000]
# CHECK-NEXT: [30.000000]
# CHECK-NEXT: [32.000000]


# CHECK: Testing 3D:
# CHECK-NEXT: [2.000000]
# CHECK-NEXT: [4.000000]
# CHECK-NEXT: [6.000000]
# CHECK-NEXT: [8.000000]
# CHECK-NEXT: [10.000000]
# CHECK-NEXT: [12.000000]
# CHECK-NEXT: [14.000000]
# CHECK-NEXT: [16.000000]
# CHECK-NEXT: [18.000000]
# CHECK-NEXT: [20.000000]
# CHECK-NEXT: [22.000000]
# CHECK-NEXT: [24.000000]
# CHECK-NEXT: [26.000000]
# CHECK-NEXT: [28.000000]
# CHECK-NEXT: [30.000000]
# CHECK-NEXT: [32.000000]


# CHECK: Testing 4D:
# CHECK-NEXT: [2.000000]
# CHECK-NEXT: [4.000000]
# CHECK-NEXT: [6.000000]
# CHECK-NEXT: [8.000000]
# CHECK-NEXT: [10.000000]
# CHECK-NEXT: [12.000000]
# CHECK-NEXT: [14.000000]
# CHECK-NEXT: [16.000000]
# CHECK-NEXT: [18.000000]
# CHECK-NEXT: [20.000000]
# CHECK-NEXT: [22.000000]
# CHECK-NEXT: [24.000000]
# CHECK-NEXT: [26.000000]
# CHECK-NEXT: [28.000000]
# CHECK-NEXT: [30.000000]
# CHECK-NEXT: [32.000000]
# CHECK-NEXT: [34.000000]
# CHECK-NEXT: [36.000000]
# CHECK-NEXT: [38.000000]
# CHECK-NEXT: [40.000000]
# CHECK-NEXT: [42.000000]
# CHECK-NEXT: [44.000000]
# CHECK-NEXT: [46.000000]
# CHECK-NEXT: [48.000000]
# CHECK-NEXT: [50.000000]
# CHECK-NEXT: [52.000000]
# CHECK-NEXT: [54.000000]
# CHECK-NEXT: [56.000000]
# CHECK-NEXT: [58.000000]
# CHECK-NEXT: [60.000000]
# CHECK-NEXT: [62.000000]
# CHECK-NEXT: [64.000000]


# CHECK: Testing 5D:
# CHECK-NEXT: [2.000000]
# CHECK-NEXT: [4.000000]
# CHECK-NEXT: [6.000000]
# CHECK-NEXT: [8.000000]
# CHECK-NEXT: [10.000000]
# CHECK-NEXT: [12.000000]
# CHECK-NEXT: [14.000000]
# CHECK-NEXT: [16.000000]
# CHECK-NEXT: [18.000000]
# CHECK-NEXT: [20.000000]
# CHECK-NEXT: [22.000000]
# CHECK-NEXT: [24.000000]
# CHECK-NEXT: [26.000000]
# CHECK-NEXT: [28.000000]
# CHECK-NEXT: [30.000000]
# CHECK-NEXT: [32.000000]
# CHECK-NEXT: [34.000000]
# CHECK-NEXT: [36.000000]
# CHECK-NEXT: [38.000000]
# CHECK-NEXT: [40.000000]
# CHECK-NEXT: [42.000000]
# CHECK-NEXT: [44.000000]
# CHECK-NEXT: [46.000000]
# CHECK-NEXT: [48.000000]
# CHECK-NEXT: [50.000000]
# CHECK-NEXT: [52.000000]
# CHECK-NEXT: [54.000000]
# CHECK-NEXT: [56.000000]
# CHECK-NEXT: [58.000000]
# CHECK-NEXT: [60.000000]
# CHECK-NEXT: [62.000000]
# CHECK-NEXT: [64.000000]
