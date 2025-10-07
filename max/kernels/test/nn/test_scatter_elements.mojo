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

from buffer import DimList
from internal_utils import TestTensor, assert_equal
from nn.gather_scatter import scatter_elements


fn test_case[
    dtype: DType,
](
    axis: Int,
    data: TestTensor[dtype, 2],
    indices: TestTensor[DType.int32, 2],
    updates: TestTensor[dtype, 2],
    output: TestTensor[dtype, 2],
) raises:
    @always_inline
    @parameter
    fn use_update[
        _dtype: DType, width: Int
    ](input_val: SIMD[_dtype, width], update_val: SIMD[_dtype, width]) -> SIMD[
        _dtype, width
    ]:
        return update_val

    test_case[dtype, use_update](axis, data, indices, updates, output)


fn test_case[
    dtype: DType,
    reduce_fn: fn[dtype: DType, width: Int] (
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing [_] -> SIMD[dtype, width],
](
    axis: Int,
    data: TestTensor[dtype, 2],
    indices: TestTensor[DType.int32, 2],
    updates: TestTensor[dtype, 2],
    output: TestTensor[dtype, 2],
) raises:
    var output_ref = output

    scatter_elements[reduce_fn](
        data.to_managed_tensor_slice(),
        indices.to_managed_tensor_slice(),
        updates.to_managed_tensor_slice(),
        axis,
        output.to_managed_tensor_slice(),
    )

    assert_equal(output, output_ref)


def main():
    fn test_scatter_ax0() raises:
        print("== test_scatter_ax0")
        var data = TestTensor[DType.float32, 2](
            DimList(3, 3), List[Float32](0, 0, 0, 0, 0, 0, 0, 0, 0)
        )
        var indices = TestTensor[DType.int32, 2](
            DimList(2, 3), List[Int32](1, 0, 2, 0, 2, 1)
        )
        var updates = TestTensor[DType.float32, 2](
            DimList(2, 3), List[Float32](1.0, 1.1, 1.2, 2.0, 2.1, 2.2)
        )
        var output_ref = TestTensor[DType.float32, 2](
            DimList(3, 3),
            List[Float32](2.0, 1.1, 0.0, 1.0, 0.0, 2.2, 0.0, 2.1, 1.2),
        )
        test_case[DType.float32](0, data, indices, updates, output_ref)

    # CHECK-LABEL: test_scatter_ax0
    # CHECK-NOT: FAIL
    test_scatter_ax0()

    fn test_scatter_ax1() raises:
        print("== test_scatter_ax1")
        var data = TestTensor[DType.float32, 2](
            DimList(1, 5), List[Float32](1, 2, 3, 4, 5)
        )
        var indices = TestTensor[DType.int32, 2](
            DimList(1, 2), List[Int32](1, 3)
        )
        var updates = TestTensor[DType.float32, 2](
            DimList(1, 2), List[Float32](1.1, 2.1)
        )
        var output_ref = TestTensor[DType.float32, 2](
            DimList(1, 5), List[Float32](1.0, 1.1, 3.0, 2.1, 5.0)
        )
        test_case[DType.float32](1, data, indices, updates, output_ref)

    # CHECK-LABEL: test_scatter_ax1
    # CHECK-NOT: FAIL
    test_scatter_ax1()

    fn test_scatter_neg_indices() raises:
        print("== test_scatter_neg_indices")
        var data = TestTensor[DType.float32, 2](
            DimList(1, 5), List[Float32](1, 2, 3, 4, 5)
        )
        var indices = TestTensor[DType.int32, 2](
            DimList(1, 2), List[Int32](1, -3)
        )
        var updates = TestTensor[DType.float32, 2](
            DimList(1, 2), List[Float32](1.1, 2.1)
        )
        var output_ref = TestTensor[DType.float32, 2](
            DimList(1, 5), List[Float32](1.0, 1.1, 2.1, 4.0, 5.0)
        )
        test_case[DType.float32](1, data, indices, updates, output_ref)

    # CHECK-LABEL: test_scatter_neg_indices
    # CHECK-NOT: FAIL
    test_scatter_neg_indices()

    fn test_scatter_reduce_max() raises:
        print("== test_scatter_reduce_max")
        var data = TestTensor[DType.float32, 2](
            DimList(1, 5), List[Float32](1, 2, 3, 4, 5)
        )
        var indices = TestTensor[DType.int32, 2](
            DimList(1, 2), List[Int32](1, 1)
        )
        var updates = TestTensor[DType.float32, 2](
            DimList(1, 2), List[Float32](1.1, 2.1)
        )
        var output_ref = TestTensor[DType.float32, 2](
            DimList(1, 5), List[Float32](1.0, 2.1, 3.0, 4.0, 5.0)
        )

        @always_inline
        @parameter
        fn _max[
            ty: DType, width: Int
        ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
            return max(v1, v2)

        test_case[DType.float32, _max](1, data, indices, updates, output_ref)

    # CHECK-LABEL: test_scatter_reduce_max
    # CHECK-NOT: FAIL
    test_scatter_reduce_max()
