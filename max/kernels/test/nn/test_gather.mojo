# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
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
        var input = LayoutTensor[
            DType.float32,
            Layout.row_major(num_rows, row_size),
            MutableAnyOrigin,
        ].stack_allocation[stack_alignment=64]()

        for i in range(num_rows):
            for j in range(row_size):
                input[i, j] = Float32(i)

        # Setup indices.
        alias num_indices = 16
        var indices = LayoutTensor[
            indices_type,
            Layout(num_indices),
            MutableAnyOrigin,
        ].stack_allocation[stack_alignment=64]()

        for i in range(num_indices):
            indices[i] = i // 2
        indices[0] = -1
        indices[1] = -num_rows

        # create output
        var output = LayoutTensor[
            DType.float32,
            Layout.row_major(num_indices, row_size),
            MutableAnyOrigin,
        ].stack_allocation[stack_alignment=64]()

        # Test gather
        alias simd_width = simd_width_of[__mlir_type.`!pop.scalar<f32>`]()

        alias output_layout = Layout.row_major[output.rank]()
        alias input_layout = Layout.row_major[input.rank]()
        alias indices_layout = Layout.row_major[indices.rank]()

        gather[axis=0](
            LayoutTensor[output.dtype, output_layout](
                output.ptr,
                RuntimeLayout[output_layout].row_major(
                    output.runtime_layout.shape.value
                ),
            ),
            LayoutTensor[input.dtype, input_layout](
                input.ptr,
                RuntimeLayout[input_layout].row_major(
                    input.runtime_layout.shape.value
                ),
            ),
            LayoutTensor[indices.dtype, indices_layout](
                indices.ptr,
                RuntimeLayout[indices_layout].row_major(
                    indices.runtime_layout.shape.value
                ),
            ),
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
        alias num_rows = 16
        alias row_size = 4

        # Setup input.
        var input = LayoutTensor[
            DType.float32,
            Layout.row_major(num_rows, row_size, 1),
            MutableAnyOrigin,
        ].stack_allocation[stack_alignment=64]()

        for i in range(num_rows):
            for j in range(row_size):
                input[i, j, 0] = Float32(i)

        # Setup indices.
        alias num_indices = 16
        var indices = LayoutTensor[
            indices_type,
            Layout.row_major(num_indices, 1),
            MutableAnyOrigin,
        ].stack_allocation[stack_alignment=64]()

        for i in range(num_indices):
            indices[i, 0] = i // 2

        # create output
        var output = LayoutTensor[
            DType.float32,
            Layout.row_major(num_indices, 1, row_size, 1),
            MutableAnyOrigin,
        ].stack_allocation[stack_alignment=64]()

        # Test gather
        alias simd_width = simd_width_of[DType.float32]()

        alias output_layout = Layout.row_major[output.rank]()
        alias input_layout = Layout.row_major[input.rank]()
        alias indices_layout = Layout.row_major[indices.rank]()

        gather[axis=0](
            LayoutTensor[output.dtype, output_layout](
                output.ptr,
                RuntimeLayout[output_layout].row_major(
                    output.runtime_layout.shape.value
                ),
            ),
            LayoutTensor[input.dtype, input_layout](
                input.ptr,
                RuntimeLayout[input_layout].row_major(
                    input.runtime_layout.shape.value
                ),
            ),
            LayoutTensor[indices.dtype, indices_layout](
                indices.ptr,
                RuntimeLayout[indices_layout].row_major(
                    indices.runtime_layout.shape.value
                ),
            ),
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
        var input = LayoutTensor[
            DType.float32, Layout.row_major(num_rows, row_size)
        ](input_stack)

        for i in range(num_rows):
            for j in range(row_size):
                input[i, j] = Float32(i)

        # Setup indices.
        # There isn't a way to represent a stack size of 0 with InlineArray
        # so we use 1 here
        var indices_stack = InlineArray[Scalar[indices_type], 1](
            uninitialized=True
        )
        var indices = LayoutTensor[indices_type, Layout(num_indices)](
            indices_stack
        )

        for i in range(num_indices):
            indices[i] = i // 2

        # create output
        var output_stack = InlineArray[Float32, num_rows * row_size](
            uninitialized=True
        )
        var output = LayoutTensor[
            DType.float32, Layout.row_major(num_indices, row_size)
        ](output_stack)

        # Test gather
        alias simd_width = simd_width_of[DType.float32]()

        alias output_layout = Layout.row_major[output.rank]()
        alias input_layout = Layout.row_major[input.rank]()
        alias indices_layout = Layout.row_major[indices.rank]()

        gather[axis=0](
            LayoutTensor[output.dtype, output_layout](
                output.ptr,
                RuntimeLayout[output_layout].row_major(
                    output.runtime_layout.shape.value.canonicalize()
                ),
            ),
            LayoutTensor[input.dtype, input_layout](
                input.ptr,
                RuntimeLayout[input_layout].row_major(
                    input.runtime_layout.shape.value.canonicalize()
                ),
            ),
            LayoutTensor[indices.dtype, indices_layout](
                indices.ptr,
                RuntimeLayout[indices_layout].row_major(
                    indices.runtime_layout.shape.value.canonicalize()
                ),
            ),
        )

    _test_gather[DType.int32]()
    _test_gather[DType.int64]()


def main():
    test_gather()
    test_gather_3d()
    test_gather_empty_indices()
