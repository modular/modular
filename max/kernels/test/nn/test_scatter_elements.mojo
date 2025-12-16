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

from memory import LegacyUnsafePointer as UnsafePointer

from nn.gather_scatter import scatter_elements
from tensor import DynamicTensor
from testing import assert_equal

from utils import IndexList


def main():
    fn test_scatter_ax0() raises:
        print("== test_scatter_ax0")

        var data_ptr = UnsafePointer[Float32].alloc(9)
        for i in range(9):
            data_ptr[i] = 0
        var data = DynamicTensor[DType.float32, 2](data_ptr, IndexList[2](3, 3))

        var indices_ptr = UnsafePointer[Int32].alloc(6)
        indices_ptr[0] = 1
        indices_ptr[1] = 0
        indices_ptr[2] = 2
        indices_ptr[3] = 0
        indices_ptr[4] = 2
        indices_ptr[5] = 1
        var indices = DynamicTensor[DType.int32, 2](
            indices_ptr, IndexList[2](2, 3)
        )

        var updates_ptr = UnsafePointer[Float32].alloc(6)
        updates_ptr[0] = 1.0
        updates_ptr[1] = 1.1
        updates_ptr[2] = 1.2
        updates_ptr[3] = 2.0
        updates_ptr[4] = 2.1
        updates_ptr[5] = 2.2
        var updates = DynamicTensor[DType.float32, 2](
            updates_ptr, IndexList[2](2, 3)
        )

        var output_ptr = UnsafePointer[Float32].alloc(9)
        var output = DynamicTensor[DType.float32, 2](
            output_ptr, IndexList[2](3, 3)
        )

        var expected = InlineArray[Float32, 9](
            Float32(2.0), 1.1, 0.0, 1.0, 0.0, 2.2, 0.0, 2.1, 1.2
        )

        @always_inline
        @parameter
        fn use_update[
            dtype: DType, width: Int
        ](
            input_val: SIMD[dtype, width], update_val: SIMD[dtype, width]
        ) -> SIMD[dtype, width]:
            return update_val

        scatter_elements[use_update](data, indices, updates, 0, output)

        for i in range(9):
            assert_equal(output_ptr[i], expected[i])

        data_ptr.free()
        indices_ptr.free()
        updates_ptr.free()
        output_ptr.free()

    # CHECK-LABEL: test_scatter_ax0
    # CHECK-NOT: FAIL
    test_scatter_ax0()

    fn test_scatter_ax1() raises:
        print("== test_scatter_ax1")

        var data_ptr = UnsafePointer[Float32].alloc(5)
        for i in range(5):
            data_ptr[i] = i + 1
        var data = DynamicTensor[DType.float32, 2](data_ptr, IndexList[2](1, 5))

        var indices_ptr = UnsafePointer[Int32].alloc(2)
        indices_ptr[0] = 1
        indices_ptr[1] = 3
        var indices = DynamicTensor[DType.int32, 2](
            indices_ptr, IndexList[2](1, 2)
        )

        var updates_ptr = UnsafePointer[Float32].alloc(2)
        updates_ptr[0] = 1.1
        updates_ptr[1] = 2.1
        var updates = DynamicTensor[DType.float32, 2](
            updates_ptr, IndexList[2](1, 2)
        )

        var output_ptr = UnsafePointer[Float32].alloc(5)
        var output = DynamicTensor[DType.float32, 2](
            output_ptr, IndexList[2](1, 5)
        )

        var expected = InlineArray[Float32, 5](Float32(1.0), 1.1, 3.0, 2.1, 5.0)

        @always_inline
        @parameter
        fn use_update[
            dtype: DType, width: Int
        ](
            input_val: SIMD[dtype, width], update_val: SIMD[dtype, width]
        ) -> SIMD[dtype, width]:
            return update_val

        scatter_elements[use_update](data, indices, updates, 1, output)

        for i in range(5):
            assert_equal(output_ptr[i], expected[i])

        data_ptr.free()
        indices_ptr.free()
        updates_ptr.free()
        output_ptr.free()

    # CHECK-LABEL: test_scatter_ax1
    # CHECK-NOT: FAIL
    test_scatter_ax1()

    fn test_scatter_neg_indices() raises:
        print("== test_scatter_neg_indices")

        var data_ptr = UnsafePointer[Float32].alloc(5)
        for i in range(5):
            data_ptr[i] = i + 1
        var data = DynamicTensor[DType.float32, 2](data_ptr, IndexList[2](1, 5))

        var indices_ptr = UnsafePointer[Int32].alloc(2)
        indices_ptr[0] = 1
        indices_ptr[1] = -3
        var indices = DynamicTensor[DType.int32, 2](
            indices_ptr, IndexList[2](1, 2)
        )

        var updates_ptr = UnsafePointer[Float32].alloc(2)
        updates_ptr[0] = 1.1
        updates_ptr[1] = 2.1
        var updates = DynamicTensor[DType.float32, 2](
            updates_ptr, IndexList[2](1, 2)
        )

        var output_ptr = UnsafePointer[Float32].alloc(5)
        var output = DynamicTensor[DType.float32, 2](
            output_ptr, IndexList[2](1, 5)
        )

        var expected = InlineArray[Float32, 5](Float32(1.0), 1.1, 2.1, 4.0, 5.0)

        @always_inline
        @parameter
        fn use_update[
            dtype: DType, width: Int
        ](
            input_val: SIMD[dtype, width], update_val: SIMD[dtype, width]
        ) -> SIMD[dtype, width]:
            return update_val

        scatter_elements[use_update](data, indices, updates, 1, output)

        for i in range(5):
            assert_equal(output_ptr[i], expected[i])

        data_ptr.free()
        indices_ptr.free()
        updates_ptr.free()
        output_ptr.free()

    # CHECK-LABEL: test_scatter_neg_indices
    # CHECK-NOT: FAIL
    test_scatter_neg_indices()

    fn test_scatter_reduce_max() raises:
        print("== test_scatter_reduce_max")

        var data_ptr = UnsafePointer[Float32].alloc(5)
        for i in range(5):
            data_ptr[i] = i + 1
        var data = DynamicTensor[DType.float32, 2](data_ptr, IndexList[2](1, 5))

        var indices_ptr = UnsafePointer[Int32].alloc(2)
        indices_ptr[0] = 1
        indices_ptr[1] = 1
        var indices = DynamicTensor[DType.int32, 2](
            indices_ptr, IndexList[2](1, 2)
        )

        var updates_ptr = UnsafePointer[Float32].alloc(2)
        updates_ptr[0] = 1.1
        updates_ptr[1] = 2.1
        var updates = DynamicTensor[DType.float32, 2](
            updates_ptr, IndexList[2](1, 2)
        )

        var output_ptr = UnsafePointer[Float32].alloc(5)
        var output = DynamicTensor[DType.float32, 2](
            output_ptr, IndexList[2](1, 5)
        )

        var expected = InlineArray[Float32, 5](Float32(1.0), 2.1, 3.0, 4.0, 5.0)

        @always_inline
        @parameter
        fn _max[
            ty: DType, width: Int
        ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
            return max(v1, v2)

        scatter_elements[_max](data, indices, updates, 1, output)

        for i in range(5):
            assert_equal(output_ptr[i], expected[i])

        data_ptr.free()
        indices_ptr.free()
        updates_ptr.free()
        output_ptr.free()

    # CHECK-LABEL: test_scatter_reduce_max
    # CHECK-NOT: FAIL
    test_scatter_reduce_max()
