# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

# Test gather_2D_input_1D_indices_axis_0.
# This test verifies that the prefetch function in `gather` passes
# compilation. The test can also be used to check the assembly to see
# if compiler generates proper SIMD instructions and unrolling.

from sys import simdwidthof

from buffer import NDBuffer
from buffer.dimlist import DimList
from nn.gather_scatter import gather

from utils.index import IndexList


# CHECK-LABEL: test_gather
fn test_gather() raises:
    print("== test_gather")

    @always_inline
    @parameter
    fn _test_gather[indices_type: DType]() raises:
        alias num_rows = 16
        alias row_size = 4

        # Setup input.
        var input = NDBuffer[
            DType.float32,
            2,
            DimList(num_rows, row_size),
        ].stack_allocation[alignment=64]()

        for i in range(num_rows):
            for j in range(row_size):
                input[IndexList[2](i, j)] = Float32(i).value

        # Setup indices.
        alias num_indices = 16
        var indices = NDBuffer[
            indices_type,
            1,
            DimList(num_indices),
        ].stack_allocation[alignment=64]()

        for i in range(num_indices):
            indices[IndexList[1](i)] = i // 2
        indices[0] = -1
        indices[1] = -num_rows

        # create output
        var output = NDBuffer[
            DType.float32,
            2,
            DimList(num_indices, row_size),
        ].stack_allocation[alignment=64]()

        # Test gather
        alias simd_width = simdwidthof[__mlir_type.`!pop.scalar<f32>`]()

        gather[axis=0](
            output.make_dims_unknown(),
            input.make_dims_unknown(),
            indices.make_dims_unknown(),
        )

        print(output[IndexList[2](0, 0)])
        print(output[IndexList[2](1, 0)])
        print(output[IndexList[2](2, 0)])
        print(output[IndexList[2](6, 0)])
        print(output[IndexList[2](15, 0)])

    # CHECK: 15.0
    # CHECK: 0.0
    # CHECK-NEXT: 1.0
    # CHECK-NEXT: 3.0
    # CHECK-NEXT: 7.0
    _test_gather[DType.int32]()
    # CHECK: 0.0
    # CHECK-NEXT: 1.0
    # CHECK-NEXT: 3.0
    # CHECK-NEXT: 7.0
    _test_gather[DType.int64]()


fn test_gather_3d() raises:
    print("== test_gather_3d\n")

    @always_inline
    @parameter
    fn _test_gather[indices_type: DType]() raises:
        alias num_rows = 16
        alias row_size = 4

        # Setup input.
        var input = NDBuffer[
            DType.float32,
            3,
            DimList(num_rows, row_size, 1),
        ].stack_allocation[alignment=64]()

        for i in range(num_rows):
            for j in range(row_size):
                input[IndexList[3](i, j, 0)] = Float32(i).value

        # Setup indices.
        alias num_indices = 16
        var indices = NDBuffer[
            indices_type,
            2,
            DimList(num_indices, 1),
        ].stack_allocation[alignment=64]()

        for i in range(num_indices):
            indices[IndexList[2](i, 0)] = i // 2

        # create output
        var output = NDBuffer[
            DType.float32,
            4,
            DimList(num_indices, 1, row_size, 1),
        ].stack_allocation[alignment=64]()

        # Test gather
        alias simd_width = simdwidthof[DType.float32]()

        gather[axis=0](
            output.make_dims_unknown(),
            input.make_dims_unknown(),
            indices.make_dims_unknown(),
        )

        print(output[IndexList[4](0, 0, 0, 0)])
        print(output[IndexList[4](2, 0, 0, 0)])
        print(output[IndexList[4](6, 0, 0, 0)])
        print(output[IndexList[4](15, 0, 0, 0)])

    # CHECK: 0.0
    # CHECK-NEXT: 1.0
    # CHECK-NEXT: 3.0
    # CHECK-NEXT: 7.0
    _test_gather[DType.int32]()
    # CHECK: 0.0
    # CHECK-NEXT: 1.0
    # CHECK-NEXT: 3.0
    # CHECK-NEXT: 7.0
    _test_gather[DType.int64]()


# CHECK-LABEL: test_gather_empty_indices
fn test_gather_empty_indices() raises:
    print("== test_gather_empty_indices")

    @always_inline
    @parameter
    fn _test_gather[indices_type: DType]() raises:
        alias num_rows = 16
        alias row_size = 4
        alias input_size = 64
        alias num_indices = 0
        alias indices_size = 0
        alias output_size = 0

        # Setup input.
        var input_stack = InlineArray[Float32, num_rows * row_size](
            uninitialized=True
        )
        var input = NDBuffer[
            DType.float32,
            2,
            DimList(num_rows, row_size),
        ](input_stack.unsafe_ptr())

        for i in range(num_rows):
            for j in range(row_size):
                input[IndexList[2](i, j)] = Float32(i).value

        # Setup indices.
        # There isn't a way to represent a stack size of 0 with InlineArray
        # so we use 1 here
        var indices_stack = InlineArray[Scalar[indices_type], 1](
            uninitialized=True
        )
        var indices = NDBuffer[indices_type, 1, DimList(num_indices)](
            indices_stack.unsafe_ptr()
        )

        for i in range(num_indices):
            indices[IndexList[1](i)] = i // 2

        # create output
        var output_stack = InlineArray[Float32, num_rows * row_size](
            uninitialized=True
        )
        var output = NDBuffer[DType.float32, 2, DimList(num_indices, row_size)](
            output_stack.unsafe_ptr()
        )

        # Test gather
        alias simd_width = simdwidthof[DType.float32]()

        gather[axis=0](
            output.make_dims_unknown(),
            input.make_dims_unknown(),
            indices.make_dims_unknown(),
        )

    _test_gather[DType.int32]()
    _test_gather[DType.int64]()


fn main() raises:
    test_gather()
    test_gather_3d()
    test_gather_empty_indices()
