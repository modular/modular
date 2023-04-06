# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s

# Test gather_2D_input_1D_indices_axis_0.
# This test verifies that the prefetch function in `gather` passes
# compilation. The test can also be used to check the assembly to see
# if compiler generates proper SIMD instructions and unrolling.

from Buffer import NDBuffer
from DType import DType
from SIMD import F32
from Gather import gather, gather_nd
from Index import StaticIntTuple
from IO import print
from List import create_dim_list
from LLCL import Runtime, OwningOutputChainPtr
from Range import range
from SIMD import SIMD
from TargetInfo import simd_width


# CHECK-LABEL: test_gather
fn test_gather():
    print("== test_gather")

    @always_inline
    fn _test_gather[indices_type: DType]():
        alias num_rows = 16
        alias row_size = 4

        # Setup input.
        var input = NDBuffer[
            2,
            create_dim_list(num_rows, row_size),
            DType.f32,
        ].aligned_stack_allocation[64]()

        for i in range(num_rows):
            for j in range(row_size):
                input[StaticIntTuple[2](i, j)] = F32(i).value

        # Setup indices.
        alias num_indices = 16
        var indices = NDBuffer[
            1,
            create_dim_list(num_indices),
            indices_type,
        ].aligned_stack_allocation[64]()

        for ii in range(num_indices):
            indices[StaticIntTuple[1](ii)] = ii // 2

        # create output
        var output = NDBuffer[
            2,
            create_dim_list(num_indices, row_size),
            DType.f32,
        ].aligned_stack_allocation[64]()

        # Test gather
        alias vector_width = simd_width[__mlir_type.`!pop.scalar<f32>`]()

        let rt = Runtime(4)
        let out_chain = OwningOutputChainPtr(rt)
        gather[
            2,
            create_dim_list(num_indices, row_size),
            2,
            create_dim_list(num_rows, row_size),
            1,
            create_dim_list(num_indices),
            DType.f32,
            indices_type,
            0,
            vector_width,
        ](output, input, indices, out_chain.borrow())
        out_chain.wait()
        out_chain.__del__()
        rt.__del__()

        print(output[StaticIntTuple[2](0, 0)])
        print(output[StaticIntTuple[2](2, 0)])
        print(output[StaticIntTuple[2](6, 0)])
        print(output[StaticIntTuple[2](15, 0)])

    # CHECK: 0.000000
    # CHECK-NEXT: 1.000000
    # CHECK-NEXT: 3.000000
    # CHECK-NEXT: 7.000000
    _test_gather[DType.si32]()
    # CHECK: 0.000000
    # CHECK-NEXT: 1.000000
    # CHECK-NEXT: 3.000000
    # CHECK-NEXT: 7.000000
    _test_gather[DType.si64]()


fn test_gather_3d():
    print("== test_gather_3d\n")

    @always_inline
    fn _test_gather[indices_type: DType]():
        alias num_rows = 16
        alias row_size = 4

        # Setup input.
        var input = NDBuffer[
            3,
            create_dim_list(num_rows, row_size, 1),
            DType.f32,
        ].aligned_stack_allocation[64]()

        for i in range(num_rows):
            for j in range(row_size):
                input[StaticIntTuple[3](i, j, 0)] = F32(i).value

        # Setup indices.
        alias num_indices = 16
        var indices = NDBuffer[
            2,
            create_dim_list(num_indices, 1),
            indices_type,
        ].aligned_stack_allocation[64]()

        for ii in range(num_indices):
            indices[StaticIntTuple[2](ii, 0)] = ii // 2

        # create output
        var output = NDBuffer[
            4,
            create_dim_list(num_indices, 1, row_size, 1),
            DType.f32,
        ].aligned_stack_allocation[64]()

        # Test gather
        alias vector_width = simd_width[__mlir_type.`!pop.scalar<f32>`]()

        let rt = Runtime(4)
        let out_chain = OwningOutputChainPtr(rt)
        gather_nd[
            4,
            create_dim_list(num_indices, 1, row_size, 1),
            3,
            create_dim_list(num_rows, row_size, 1),
            2,
            create_dim_list(num_indices, 1),
            DType.f32,
            indices_type,
            0,
            vector_width,
        ](output, input, indices, out_chain.borrow())
        out_chain.wait()
        out_chain.__del__()
        rt.__del__()

        print(output[StaticIntTuple[4](0, 0, 0, 0)])
        print(output[StaticIntTuple[4](2, 0, 0, 0)])
        print(output[StaticIntTuple[4](6, 0, 0, 0)])
        print(output[StaticIntTuple[4](15, 0, 0, 0)])

    # CHECK: 0.000000
    # CHECK-NEXT: 1.000000
    # CHECK-NEXT: 3.000000
    # CHECK-NEXT: 7.000000
    _test_gather[DType.si32]()
    # CHECK: 0.000000
    # CHECK-NEXT: 1.000000
    # CHECK-NEXT: 3.000000
    # CHECK-NEXT: 7.000000
    _test_gather[DType.si64]()


# CHECK-LABEL: test_gather_empty_indices
fn test_gather_empty_indices():
    print("== test_gather_empty_indices")

    @always_inline
    fn _test_gather[indices_type: DType]():
        alias num_rows = 16
        alias row_size = 4
        alias input_size = 64
        alias num_indices = 0
        alias indices_size = 0
        alias output_size = 0

        # Setup input.
        var input = NDBuffer[
            2,
            create_dim_list(num_rows, row_size),
            DType.f32,
        ].aligned_stack_allocation[input_size]()

        for i in range(num_rows):
            for j in range(row_size):
                input[StaticIntTuple[2](i, j)] = F32(i).value

        # Setup indices.
        var indices = NDBuffer[
            1,
            create_dim_list(num_indices),
            indices_type,
        ].aligned_stack_allocation[indices_size]()

        for ii in range(num_indices):
            indices[StaticIntTuple[1](ii)] = ii // 2

        # create output
        var output = NDBuffer[
            2,
            create_dim_list(num_indices, row_size),
            DType.f32,
        ].aligned_stack_allocation[output_size]()

        # Test gather
        alias vector_width = simd_width[__mlir_type.`!pop.scalar<f32>`]()

        let rt = Runtime(4)
        let out_chain = OwningOutputChainPtr(rt)
        gather[
            2,
            create_dim_list(num_indices, row_size),
            2,
            create_dim_list(num_rows, row_size),
            1,
            create_dim_list(num_indices),
            DType.f32,
            indices_type,
            0,
            vector_width,
        ](output, input, indices, out_chain.borrow())
        out_chain.wait()
        out_chain.__del__()
        rt.__del__()

    _test_gather[DType.si32]()
    _test_gather[DType.si64]()


fn main():
    test_gather()
    test_gather_3d()
    test_gather_empty_indices()
