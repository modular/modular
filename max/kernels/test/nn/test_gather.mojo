# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# Test gather_2D_input_1D_indices_axis_0.
# This test verifies that the prefetch function in `gather` passes
# compilation. The test can also be used to check the assembly to see
# if compiler generates proper SIMD instructions and unrolling.

from sys import simd_width_of

from layout._layout import row_major
from layout._tile_tensor import TileTensor
from nn.gather_scatter import gather


# CHECK-LABEL: test_gather
fn test_gather() raises:
    print("== test_gather")

    @always_inline
    @parameter
    fn _test_gather[indices_type: DType]() raises:
        comptime num_rows = 16
        comptime row_size = 4

        # Setup input.
        var input_stack = InlineArray[Float32, num_rows * row_size](
            uninitialized=True
        )
        var input = TileTensor(input_stack, row_major[num_rows, row_size]())

        for i in range(num_rows):
            for j in range(row_size):
                input[i, j] = Float32(i)

        # Setup indices.
        comptime num_indices = 16
        var indices_stack = InlineArray[Scalar[indices_type], num_indices](
            uninitialized=True
        )
        var indices = TileTensor(indices_stack, row_major[num_indices]())

        for i in range(num_indices):
            indices[i] = Scalar[indices_type](i // 2)
        indices[0] = -1
        indices[1] = -num_rows

        # create output
        var output_stack = InlineArray[Float32, num_indices * row_size](
            uninitialized=True
        )
        var output = TileTensor(
            output_stack, row_major[num_indices, row_size]()
        )

        # Test gather
        comptime simd_width = simd_width_of[__mlir_type.`!pop.scalar<f32>`]()

        gather[axis=0](
            output.make_dynamic[DType.int64](),
            input.make_dynamic[DType.int64](),
            indices.make_dynamic[DType.int64](),
        )

        print(output[0, 0])
        print(output[1, 0])
        print(output[2, 0])
        print(output[6, 0])
        print(output[15, 0])

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
        comptime num_rows = 16
        comptime row_size = 4

        # Setup input.
        var input_stack = InlineArray[Float32, num_rows * row_size * 1](
            uninitialized=True
        )
        var input = TileTensor(input_stack, row_major[num_rows, row_size, 1]())

        for i in range(num_rows):
            for j in range(row_size):
                input[i, j, 0] = Float32(i)

        # Setup indices.
        comptime num_indices = 16
        var indices_stack = InlineArray[Scalar[indices_type], num_indices * 1](
            uninitialized=True
        )
        var indices = TileTensor(indices_stack, row_major[num_indices, 1]())

        for i in range(num_indices):
            indices[i, 0] = Scalar[indices_type](i // 2)

        # create output
        var output_stack = InlineArray[Float32, num_indices * 1 * row_size * 1](
            uninitialized=True
        )
        var output = TileTensor(
            output_stack, row_major[num_indices, 1, row_size, 1]()
        )

        # Test gather
        comptime simd_width = simd_width_of[DType.float32]()

        gather[axis=0](
            output.make_dynamic[DType.int64](),
            input.make_dynamic[DType.int64](),
            indices.make_dynamic[DType.int64](),
        )

        print(output[0, 0, 0, 0])
        print(output[2, 0, 0, 0])
        print(output[6, 0, 0, 0])
        print(output[15, 0, 0, 0])

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
        comptime num_rows = 16
        comptime row_size = 4
        comptime num_indices = 0

        # Setup input.
        var input_stack = InlineArray[Float32, num_rows * row_size](
            uninitialized=True
        )
        var input = TileTensor(input_stack, row_major[num_rows, row_size]())

        for i in range(num_rows):
            for j in range(row_size):
                input[i, j] = Float32(i)

        # Setup indices.
        # There isn't a way to represent a stack size of 0 with InlineArray
        # so we use 1 here
        var indices_stack = InlineArray[Scalar[indices_type], 1](
            uninitialized=True
        )
        var indices = TileTensor(indices_stack, row_major[num_indices]())

        for i in range(num_indices):
            indices[i] = Scalar[indices_type](i // 2)

        # create output
        var output_stack = InlineArray[Float32, num_rows * row_size](
            uninitialized=True
        )
        var output = TileTensor(
            output_stack, row_major[num_indices, row_size]()
        )

        # Test gather
        comptime simd_width = simd_width_of[DType.float32]()

        gather[axis=0](
            output.make_dynamic[DType.int64](),
            input.make_dynamic[DType.int64](),
            indices.make_dynamic[DType.int64](),
        )

    _test_gather[DType.int32]()
    _test_gather[DType.int64]()


def main():
    test_gather()
    test_gather_3d()
    test_gather_empty_indices()
