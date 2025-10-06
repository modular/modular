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


from layout import LayoutTensor, Layout, RuntimeLayout, UNKNOWN_VALUE
from nn.gather_scatter import _gather_nd_impl, gather_nd_shape

from utils import IndexList


# CHECK-LABEL: test_gather_nd
fn main():
    """Note: Examples 1-5 are from.

    https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND
    """

    print("test_gather_nd")

    fn test_gather_nd_eg1() raises:
        # Example 1
        alias batch_dims = 0
        alias data_rank = 2
        alias data_type = DType.int32
        var data_stack = InlineArray[Scalar[data_type], 4](uninitialized=True)
        alias data_layout = Layout.row_major[data_rank]()
        var data = LayoutTensor[data_type, Layout.row_major(2, 2)](data_stack)

        data[0, 0] = 0
        data[0, 1] = 1
        data[1, 0] = 2
        data[1, 1] = 3

        alias indices_rank = 2
        alias indices_layout = Layout.row_major[indices_rank]()
        var indices_stack = InlineArray[Int64, 4](uninitialized=True)
        var indices = LayoutTensor[DType.int64, Layout.row_major(2, 2)](
            indices_stack
        )

        indices[0, 0] = 0
        indices[0, 1] = 0
        indices[1, 0] = 1
        indices[1, 1] = 1

        alias output_rank = 1
        var output_shape = gather_nd_shape[
            output_rank, data_type, DType.int64, batch_dims
        ](
            LayoutTensor[data.dtype, data_layout](
                data.ptr,
                RuntimeLayout[data_layout].row_major(
                    data.runtime_layout.shape.value.canonicalize()
                ),
            ),
            LayoutTensor[indices.dtype, indices_layout](
                indices.ptr,
                RuntimeLayout[indices_layout].row_major(
                    indices.runtime_layout.shape.value.canonicalize()
                ),
            ),
        )
        print("Output shape: ", output_shape)

        var output_data_data = InlineArray[Scalar[data_type], 2](
            uninitialized=True
        )
        alias output_layout = Layout.row_major[output_rank]()
        var output_data_buffer = LayoutTensor[data_type, output_layout](
            output_data_data.unsafe_ptr(),
            RuntimeLayout[output_layout].row_major(output_shape),
        )
        _gather_nd_impl[batch_dims](
            LayoutTensor[data.dtype, data_layout](
                data.ptr,
                RuntimeLayout[data_layout].row_major(
                    data.runtime_layout.shape.value.canonicalize()
                ),
            ),
            LayoutTensor[indices.dtype, indices_layout](
                indices.ptr,
                RuntimeLayout[indices_layout].row_major(
                    indices.runtime_layout.shape.value.canonicalize()
                ),
            ),
            output_data_buffer,
        )
        print(
            "Output buffer:", output_data_buffer[0], ",", output_data_buffer[1]
        )

    fn test_gather_nd_eg2() raises:
        # Example 2
        alias batch_dims = 0
        alias data_rank = 2
        alias data_type = DType.int8
        alias data_layout = Layout.row_major[data_rank]()
        var data_stack = InlineArray[Scalar[data_type], 4](uninitialized=True)
        var data = LayoutTensor[data_type, Layout.row_major(2, 2)](data_stack)

        data[0, 0] = 0
        data[0, 1] = 1
        data[1, 0] = 2
        data[1, 1] = 3

        alias indices_rank = 2
        alias indices_layout = Layout.row_major[indices_rank]()
        var indices_stack = InlineArray[Int64, 2](uninitialized=True)
        var indices = LayoutTensor[DType.int64, Layout.row_major(2, 1)](
            indices_stack
        )

        indices[0, 0] = 1
        indices[1, 0] = 0

        alias output_rank = 2
        var output_shape = gather_nd_shape[
            output_rank,
            data_type,
            DType.int64,
            batch_dims,
        ](
            LayoutTensor[data.dtype, data_layout](
                data.ptr,
                RuntimeLayout[data_layout].row_major(
                    data.runtime_layout.shape.value.canonicalize()
                ),
            ),
            LayoutTensor[indices.dtype, indices_layout](
                indices.ptr,
                RuntimeLayout[indices_layout].row_major(
                    indices.runtime_layout.shape.value.canonicalize()
                ),
            ),
        )
        print("Output shape: ", output_shape)

        var output_data_data = InlineArray[Scalar[data_type], 4](
            uninitialized=True
        )
        alias output_layout = Layout.row_major[output_rank]()
        var output_data_buffer = LayoutTensor[data_type, output_layout](
            output_data_data.unsafe_ptr(),
            RuntimeLayout[output_layout].row_major(output_shape),
        )
        _gather_nd_impl[batch_dims](
            LayoutTensor[data.dtype, data_layout](
                data.ptr,
                RuntimeLayout[data_layout].row_major(
                    data.runtime_layout.shape.value.canonicalize()
                ),
            ),
            LayoutTensor[indices.dtype, indices_layout](
                indices.ptr,
                RuntimeLayout[indices_layout].row_major(
                    indices.runtime_layout.shape.value.canonicalize()
                ),
            ),
            output_data_buffer,
        )
        print(
            "Output buffer:",
            output_data_buffer[0, 0],
            ",",
            output_data_buffer[0, 1],
            ",",
            output_data_buffer[1, 0],
            ",",
            output_data_buffer[1, 1],
        )

    fn test_gather_nd_eg3() raises:
        # Example 3
        alias batch_dims = 0
        alias data_rank = 3
        alias data_type = DType.float32
        alias data_layout = Layout.row_major[data_rank]()
        var data_stack = InlineArray[Scalar[data_type], 8](uninitialized=True)
        var data = LayoutTensor[data_type, Layout.row_major(2, 2, 2)](
            data_stack
        )

        data[0, 0, 0] = 0
        data[0, 0, 1] = 1
        data[0, 1, 0] = 2
        data[0, 1, 1] = 3
        data[1, 0, 0] = 4
        data[1, 0, 1] = 5
        data[1, 1, 0] = 6
        data[1, 1, 1] = 7

        alias indices_rank = 2
        alias indices_layout = Layout.row_major[indices_rank]()
        var indices_stack = InlineArray[Int64, 4](uninitialized=True)
        var indices = LayoutTensor[DType.int64, Layout.row_major(2, 2)](
            indices_stack
        )

        indices[0, 0] = 0
        indices[0, 1] = 1
        indices[1, 0] = 1
        indices[1, 1] = 0

        alias output_rank = 2
        var output_shape = gather_nd_shape[
            output_rank, data_type, DType.int64, batch_dims
        ](
            LayoutTensor[data.dtype, data_layout](
                data.ptr,
                RuntimeLayout[data_layout].row_major(
                    data.runtime_layout.shape.value.canonicalize()
                ),
            ),
            LayoutTensor[indices.dtype, indices_layout](
                indices.ptr,
                RuntimeLayout[indices_layout].row_major(
                    indices.runtime_layout.shape.value.canonicalize()
                ),
            ),
        )
        print("Output shape: ", output_shape)

        var output_data_data = InlineArray[Scalar[data_type], 4](
            uninitialized=True
        )
        alias output_layout = Layout.row_major[output_rank]()
        var output_data_buffer = LayoutTensor[data_type, output_layout](
            output_data_data.unsafe_ptr(),
            RuntimeLayout[output_layout].row_major(output_shape),
        )
        _gather_nd_impl[batch_dims](
            LayoutTensor[data.dtype, data_layout](
                data.ptr,
                RuntimeLayout[data_layout].row_major(
                    data.runtime_layout.shape.value.canonicalize()
                ),
            ),
            LayoutTensor[indices.dtype, indices_layout](
                indices.ptr,
                RuntimeLayout[indices_layout].row_major(
                    indices.runtime_layout.shape.value.canonicalize()
                ),
            ),
            output_data_buffer,
        )
        print(
            "Output buffer:",
            output_data_buffer[0, 0],
            ",",
            output_data_buffer[0, 1],
            ",",
            output_data_buffer[1, 0],
            ",",
            output_data_buffer[1, 1],
        )

    fn test_gather_nd_eg4() raises:
        # Example 4
        alias batch_dims = 0
        alias data_rank = 3
        alias data_type = DType.int8
        alias data_layout = Layout.row_major[data_rank]()
        var data_stack = InlineArray[Scalar[data_type], 8](uninitialized=True)
        var data = LayoutTensor[data_type, Layout.row_major(2, 2, 2)](
            data_stack
        )

        data[0, 0, 0] = 0
        data[0, 0, 1] = 1
        data[0, 1, 0] = 2
        data[0, 1, 1] = 3
        data[1, 0, 0] = 4
        data[1, 0, 1] = 5
        data[1, 1, 0] = 6
        data[1, 1, 1] = 7

        alias indices_rank = 3
        alias indices_layout = Layout.row_major[indices_rank]()
        var indices_stack = InlineArray[Int64, 4](uninitialized=True)
        var indices = LayoutTensor[DType.int64, Layout.row_major(2, 1, 2)](
            indices_stack
        )

        indices[0, 0, 0] = 0
        indices[0, 0, 1] = 1
        indices[1, 0, 0] = 1
        indices[1, 0, 1] = 0

        alias output_rank = 3
        var output_shape = gather_nd_shape[
            output_rank, data_type, DType.int64, batch_dims
        ](
            LayoutTensor[data.dtype, data_layout](
                data.ptr,
                RuntimeLayout[data_layout].row_major(
                    data.runtime_layout.shape.value.canonicalize()
                ),
            ),
            LayoutTensor[indices.dtype, indices_layout](
                indices.ptr,
                RuntimeLayout[indices_layout].row_major(
                    indices.runtime_layout.shape.value.canonicalize()
                ),
            ),
        )
        print("Output shape: ", output_shape)

        var output_data_data = InlineArray[Scalar[data_type], 4](
            uninitialized=True
        )
        alias output_layout = Layout.row_major[output_rank]()
        var output_data_buffer = LayoutTensor[data_type, output_layout](
            output_data_data.unsafe_ptr(),
            RuntimeLayout[output_layout].row_major(output_shape),
        )
        _gather_nd_impl[batch_dims](
            LayoutTensor[data.dtype, data_layout](
                data.ptr,
                RuntimeLayout[data_layout].row_major(
                    data.runtime_layout.shape.value.canonicalize()
                ),
            ),
            LayoutTensor[indices.dtype, indices_layout](
                indices.ptr,
                RuntimeLayout[indices_layout].row_major(
                    indices.runtime_layout.shape.value.canonicalize()
                ),
            ),
            output_data_buffer,
        )
        print(
            "Output buffer:",
            output_data_buffer[0, 0, 0],
            ",",
            output_data_buffer[0, 0, 1],
            ",",
            output_data_buffer[1, 0, 0],
            ",",
            output_data_buffer[1, 0, 1],
        )

    fn test_gather_nd_eg5() raises:
        # Example 5
        alias batch_dims = 1
        alias data_rank = 3
        alias data_type = DType.int32
        alias data_layout = Layout.row_major[data_rank]()
        var data_stack = InlineArray[Scalar[data_type], 8](uninitialized=True)
        var data = LayoutTensor[data_type, Layout.row_major(2, 2, 2)](
            data_stack
        )

        data[0, 0, 0] = 0
        data[0, 0, 1] = 1
        data[0, 1, 0] = 2
        data[0, 1, 1] = 3
        data[1, 0, 0] = 4
        data[1, 0, 1] = 5
        data[1, 1, 0] = 6
        data[1, 1, 1] = 7

        alias indices_rank = 2
        alias indices_layout = Layout.row_major[indices_rank]()
        var indices_stack = InlineArray[Int64, 2](uninitialized=True)
        var indices = LayoutTensor[DType.int64, Layout.row_major(2, 1)](
            indices_stack
        )

        indices[0, 0] = 1
        indices[1, 0] = 0

        alias output_rank = 2
        var output_shape = gather_nd_shape[
            output_rank, data_type, DType.int64, batch_dims
        ](
            LayoutTensor[data.dtype, data_layout](
                data.ptr,
                RuntimeLayout[data_layout].row_major(
                    data.runtime_layout.shape.value.canonicalize()
                ),
            ),
            LayoutTensor[indices.dtype, indices_layout](
                indices.ptr,
                RuntimeLayout[indices_layout].row_major(
                    indices.runtime_layout.shape.value.canonicalize()
                ),
            ),
        )
        print("Output shape: ", output_shape)

        var output_data_data = InlineArray[Scalar[data_type], 4](
            uninitialized=True
        )
        alias output_layout = Layout.row_major[output_rank]()
        var output_data_buffer = LayoutTensor[data_type, output_layout](
            output_data_data.unsafe_ptr(),
            RuntimeLayout[output_layout].row_major(output_shape),
        )
        _gather_nd_impl[batch_dims](
            LayoutTensor[data.dtype, data_layout](
                data.ptr,
                RuntimeLayout[data_layout].row_major(
                    data.runtime_layout.shape.value.canonicalize()
                ),
            ),
            LayoutTensor[indices.dtype, indices_layout](
                indices.ptr,
                RuntimeLayout[indices_layout].row_major(
                    indices.runtime_layout.shape.value.canonicalize()
                ),
            ),
            output_data_buffer,
        )
        print(
            "Output buffer:",
            output_data_buffer[0, 0],
            ",",
            output_data_buffer[0, 1],
            ",",
            output_data_buffer[1, 0],
            ",",
            output_data_buffer[1, 1],
        )

    fn test_gather_nd_eg6() raises:
        # Example 6
        alias batch_dims = 2
        alias data_rank = 3
        alias data_type = DType.int8
        alias data_layout = Layout.row_major[data_rank]()
        var data_stack = InlineArray[Scalar[data_type], 2 * 3 * 4](
            uninitialized=True
        )
        var data = LayoutTensor[data_type, Layout.row_major(2, 3, 4)](
            data_stack
        )

        data[0, 0, 0] = 1
        data[0, 0, 1] = 2
        data[0, 0, 2] = 3
        data[0, 0, 3] = 4

        data[0, 1, 0] = 5
        data[0, 1, 1] = 6
        data[0, 1, 2] = 7
        data[0, 1, 3] = 8

        data[0, 2, 0] = 9
        data[0, 2, 1] = 10
        data[0, 2, 2] = 11
        data[0, 2, 3] = 12

        data[1, 0, 0] = 13
        data[1, 0, 1] = 14
        data[1, 0, 2] = 15
        data[1, 0, 3] = 16

        data[1, 1, 0] = 17
        data[1, 1, 1] = 18
        data[1, 1, 2] = 19
        data[1, 1, 3] = 20

        data[1, 2, 0] = 21
        data[1, 2, 1] = 22
        data[1, 2, 2] = 23
        data[1, 2, 3] = 24

        alias indices_rank = 4
        alias indices_layout = Layout.row_major[indices_rank]()
        var indices_stack = InlineArray[Int64, 2 * 3](uninitialized=True)
        var indices = LayoutTensor[DType.int64, Layout.row_major(2, 3, 1, 1)](
            indices_stack
        )

        indices[0, 0, 0, 0] = 1
        indices[0, 1, 0, 0] = 0
        indices[0, 2, 0, 0] = 2
        indices[1, 0, 0, 0] = 0
        indices[1, 1, 0, 0] = 2
        indices[1, 2, 0, 0] = 2

        alias output_rank = 3
        var output_shape = gather_nd_shape[
            output_rank, data_type, DType.int64, batch_dims
        ](
            LayoutTensor[data.dtype, data_layout](
                data.ptr,
                RuntimeLayout[data_layout].row_major(
                    data.runtime_layout.shape.value.canonicalize()
                ),
            ),
            LayoutTensor[indices.dtype, indices_layout](
                indices.ptr,
                RuntimeLayout[indices_layout].row_major(
                    indices.runtime_layout.shape.value.canonicalize()
                ),
            ),
        )
        print("Output shape: ", output_shape)

        var output_data_data = InlineArray[Scalar[data_type], 6](
            uninitialized=True
        )
        alias output_layout = Layout.row_major[output_rank]()
        var output_data_buffer = LayoutTensor[data_type, output_layout](
            output_data_data.unsafe_ptr(),
            RuntimeLayout[output_layout].row_major(output_shape),
        )
        _gather_nd_impl[batch_dims](
            LayoutTensor[data.dtype, data_layout](
                data.ptr,
                RuntimeLayout[data_layout].row_major(
                    data.runtime_layout.shape.value.canonicalize()
                ),
            ),
            LayoutTensor[indices.dtype, indices_layout](
                indices.ptr,
                RuntimeLayout[indices_layout].row_major(
                    indices.runtime_layout.shape.value.canonicalize()
                ),
            ),
            output_data_buffer,
        )
        print(
            "Output buffer:",
            output_data_buffer[0, 0, 0],
            ",",
            output_data_buffer[0, 1, 0],
            ",",
            output_data_buffer[0, 2, 0],
            ",",
            output_data_buffer[1, 0, 0],
            ",",
            output_data_buffer[1, 1, 0],
            ",",
            output_data_buffer[1, 2, 0],
        )

    fn test_gather_nd_eg7() raises:
        # Example 4
        alias batch_dims = 0
        alias data_rank = 3
        alias data_type = DType.int8
        var data_stack = InlineArray[Scalar[data_type], 8](uninitialized=True)
        alias data_layout = Layout.row_major[data_rank]()
        var data = LayoutTensor[data_type, Layout.row_major(2, 2, 2)](
            data_stack
        )

        data[0, 0, 0] = 0
        data[0, 0, 1] = 1
        data[0, 1, 0] = 2
        data[0, 1, 1] = 3
        data[1, 0, 0] = 4
        data[1, 0, 1] = 5
        data[1, 1, 0] = 6
        data[1, 1, 1] = 7

        alias indices_rank = 3
        alias indices_layout = Layout.row_major[indices_rank]()
        var indices_stack = InlineArray[Int64, 2](uninitialized=True)
        var indices = LayoutTensor[DType.int64, Layout.row_major(2, 1, 1)](
            indices_stack
        )

        indices[0, 0, 0] = 0
        indices[1, 0, 0] = 1

        alias output_rank = 4
        alias output_layout = Layout.row_major[output_rank]()
        var output_shape = gather_nd_shape[
            output_rank, data_type, DType.int64, batch_dims
        ](
            LayoutTensor[data.dtype, data_layout](
                data.ptr,
                RuntimeLayout[data_layout].row_major(
                    data.runtime_layout.shape.value.canonicalize()
                ),
            ),
            LayoutTensor[indices.dtype, indices_layout](
                indices.ptr,
                RuntimeLayout[indices_layout].row_major(
                    indices.runtime_layout.shape.value.canonicalize()
                ),
            ),
        )
        print("Output shape: ", output_shape)

        var output_data_data = InlineArray[Scalar[data_type], 8](
            uninitialized=True
        )
        var output_data_buffer = LayoutTensor[data_type, output_layout](
            output_data_data.unsafe_ptr(),
            RuntimeLayout[output_layout].row_major(output_shape),
        )
        _gather_nd_impl[batch_dims](
            LayoutTensor[data.dtype, data_layout](
                data.ptr,
                RuntimeLayout[data_layout].row_major(
                    data.runtime_layout.shape.value.canonicalize()
                ),
            ),
            LayoutTensor[indices.dtype, indices_layout](
                indices.ptr,
                RuntimeLayout[indices_layout].row_major(
                    indices.runtime_layout.shape.value.canonicalize()
                ),
            ),
            output_data_buffer,
        )

        print(
            "Output buffer:",
            output_data_buffer[0, 0, 0, 0],
            ",",
            output_data_buffer[0, 0, 0, 1],
            ",",
            output_data_buffer[0, 0, 1, 0],
            ",",
            output_data_buffer[0, 0, 1, 1],
            ",",
            output_data_buffer[1, 0, 0, 0],
            ",",
            output_data_buffer[1, 0, 0, 1],
            ",",
            output_data_buffer[1, 0, 1, 0],
            ",",
            output_data_buffer[1, 0, 1, 1],
        )

    fn test_gather_nd_eg8() raises:
        # Example 2
        alias batch_dims = 0
        alias data_rank = 2
        alias data_type = DType.int8
        alias data_layout = Layout.row_major[data_rank]()
        var data_stack = InlineArray[Scalar[data_type], 6](uninitialized=True)
        var data = LayoutTensor[data_type, Layout.row_major(2, 3)](data_stack)

        data[0, 0] = 0
        data[0, 1] = 1
        data[0, 2] = 2
        data[1, 0] = 3
        data[1, 1] = 4
        data[1, 2] = 5

        alias indices_rank = 2
        alias indices_layout = Layout.row_major[indices_rank]()
        var indices_stack = InlineArray[Int64, 2](uninitialized=True)
        var indices = LayoutTensor[DType.int64, Layout.row_major(2, 1)](
            indices_stack
        )

        indices[0, 0] = 1
        indices[1, 0] = 0

        alias output_rank = 2
        var output_shape = gather_nd_shape[
            output_rank,
            data_type,
            DType.int64,
            batch_dims,
        ](
            LayoutTensor[data.dtype, data_layout](
                data.ptr,
                RuntimeLayout[data_layout].row_major(
                    data.runtime_layout.shape.value.canonicalize()
                ),
            ),
            LayoutTensor[indices.dtype, indices_layout](
                indices.ptr,
                RuntimeLayout[indices_layout].row_major(
                    indices.runtime_layout.shape.value.canonicalize()
                ),
            ),
        )
        print("Output shape: ", output_shape)

        alias output_layout = Layout.row_major[output_rank]()
        var output_data_data = InlineArray[Scalar[data_type], 6](
            uninitialized=True
        )
        var output_data_buffer = LayoutTensor[data_type, output_layout](
            output_data_data.unsafe_ptr(),
            RuntimeLayout[output_layout].row_major(output_shape),
        )
        _gather_nd_impl[batch_dims](
            LayoutTensor[data.dtype, data_layout](
                data.ptr,
                RuntimeLayout[data_layout].row_major(
                    data.runtime_layout.shape.value.canonicalize()
                ),
            ),
            LayoutTensor[indices.dtype, indices_layout](
                indices.ptr,
                RuntimeLayout[indices_layout].row_major(
                    indices.runtime_layout.shape.value.canonicalize()
                ),
            ),
            output_data_buffer,
        )
        print(
            "Output buffer:",
            output_data_buffer[0, 0],
            ",",
            output_data_buffer[0, 1],
            ",",
            output_data_buffer[0, 2],
            ",",
            output_data_buffer[1, 0],
            ",",
            output_data_buffer[1, 1],
            ",",
            output_data_buffer[1, 2],
        )

    try:
        # CHECK: Output shape:  (2,)
        # CHECK: Output buffer: 0 , 3
        test_gather_nd_eg1()
        # CHECK: Output shape:  (2, 2)
        # CHECK: Output buffer: 2 , 3 , 0 , 1
        test_gather_nd_eg2()
        # CHECK: Output shape:  (2, 2)
        # CHECK: Output buffer: 2.0 , 3.0 , 4.0 , 5.0
        test_gather_nd_eg3()
        # CHECK: Output shape:  (2, 1, 2)
        # CHECK: Output buffer: 2 , 3 , 4 , 5
        test_gather_nd_eg4()
        # CHECK: Output shape:  (2, 2)
        # CHECK: Output buffer: 2 , 3 , 4 , 5
        test_gather_nd_eg5()
        # CHECK: Output shape:  (2, 3, 1)
        # CHECK: Output buffer: 2 , 5 , 11 , 13 , 19 , 23
        test_gather_nd_eg6()
        # CHECK: Output shape:  (2, 1, 2, 2)
        # CHECK: Output buffer: 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7
        test_gather_nd_eg7()
        # CHECK: Output shape:  (2, 3)
        # CHECK: Output buffer: 3 , 4 , 5 , 0 , 1 , 2
        test_gather_nd_eg8()
    except e:
        print(e)
