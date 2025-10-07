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
from nn.gather_scatter import scatter_nd_generator


@always_inline
@parameter
fn use_update[
    dtype: DType, width: Int, //
](input_val: SIMD[dtype, width], update_val: SIMD[dtype, width]) -> SIMD[
    dtype, width
]:
    return update_val


fn test_case[
    dtype: DType,
](
    data: TestTensor[dtype, 3],
    indices: TestTensor[DType.int64, 2],
    updates: TestTensor[dtype, 3],
    output: TestTensor[dtype, 3],
) raises:
    @always_inline
    @parameter
    fn use_update[
        _dtype: DType, width: Int
    ](input_val: SIMD[_dtype, width], update_val: SIMD[_dtype, width]) -> SIMD[
        _dtype, width
    ]:
        return update_val

    test_case[dtype, use_update](data, indices, updates, output)


fn test_case[
    dtype: DType,
    reduce_fn: fn[dtype: DType, width: Int] (
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing [_] -> SIMD[dtype, width],
](
    data: TestTensor[dtype, 3],
    indices: TestTensor[DType.int64, 2],
    updates: TestTensor[dtype, 3],
    output: TestTensor[dtype, 3],
) raises:
    var output_ref = output

    # Note: This is for the specific set of examples
    #      (due to _to_ndbuffer[] parameters).
    # last example 3,2,2,3 ; original: 3,2,3,3
    scatter_nd_generator[dtype, DType.int64, False, reduce_fn=reduce_fn](
        data.to_managed_tensor_slice().to_layout_tensor(),
        indices.to_managed_tensor_slice().to_layout_tensor(),
        updates.to_managed_tensor_slice().to_layout_tensor(),
        output.to_managed_tensor_slice().to_layout_tensor(),
    )

    assert_equal(output, output_ref)


def main():
    fn test_scatternd() raises:
        print("== test_scatternd")
        var data = TestTensor[DType.float32, 3](
            DimList(4, 4, 4),
            List[Float32](
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
            ),
        )

        var indices = TestTensor[DType.int64, 2](
            DimList(2, 1), List[Int64](0, 2)
        )

        var updates = TestTensor[DType.float32, 3](
            DimList(2, 4, 4),
            List[Float32](
                5,
                5,
                5,
                5,
                6,
                6,
                6,
                6,
                7,
                7,
                7,
                7,
                8,
                8,
                8,
                8,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                4,
                4,
                4,
                4,
            ),
        )

        var output_ref = TestTensor[DType.float32, 3](
            DimList(4, 4, 4),
            List[Float32](
                5,
                5,
                5,
                5,
                6,
                6,
                6,
                6,
                7,
                7,
                7,
                7,
                8,
                8,
                8,
                8,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                4,
                4,
                4,
                4,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
            ),
        )

        test_case[DType.float32](
            data,
            indices,
            updates,
            output_ref,
        )

    test_scatternd()

    fn test_scatternd_add() raises:
        print("== test_scatternd_add")
        var data = TestTensor[DType.float32, 3](
            DimList(4, 4, 4),
            List[Float32](
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
            ),
        )

        var indices = TestTensor[DType.int64, 2](
            DimList(2, 1), List[Int64](0, 0)
        )

        var updates = TestTensor[DType.float32, 3](
            DimList(2, 4, 4),
            List[Float32](
                5,
                5,
                5,
                5,
                6,
                6,
                6,
                6,
                7,
                7,
                7,
                7,
                8,
                8,
                8,
                8,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                4,
                4,
                4,
                4,
            ),
        )

        var output_ref = TestTensor[DType.float32, 3](
            DimList(4, 4, 4),
            List[Float32](
                7,
                8,
                9,
                10,
                13,
                14,
                15,
                16,
                18,
                17,
                16,
                15,
                16,
                15,
                14,
                13,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
            ),
        )

        @always_inline
        @parameter
        fn _add[
            ty: DType, width: Int
        ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
            return v1 + v2

        test_case[DType.float32, _add](data, indices, updates, output_ref)

    test_scatternd_add()

    fn test_scatternd_max() raises:
        print("== test_scatternd_max")
        var data = TestTensor[DType.float32, 3](
            DimList(4, 4, 4),
            List[Float32](
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
            ),
        )

        var indices = TestTensor[DType.int64, 2](
            DimList(2, 1), List[Int64](0, 0)
        )

        var updates = TestTensor[DType.float32, 3](
            DimList(2, 4, 4),
            List[Float32](
                5,
                5,
                5,
                5,
                6,
                6,
                6,
                6,
                7,
                7,
                7,
                7,
                8,
                8,
                8,
                8,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                4,
                4,
                4,
                4,
            ),
        )

        var output_ref = TestTensor[DType.float32, 3](
            DimList(4, 4, 4),
            List[Float32](
                5,
                5,
                5,
                5,
                6,
                6,
                7,
                8,
                8,
                7,
                7,
                7,
                8,
                8,
                8,
                8,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
            ),
        )

        @always_inline
        @parameter
        fn _max[
            ty: DType, width: Int
        ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
            return max(v1, v2)

        test_case[DType.float32, _max](data, indices, updates, output_ref)

    test_scatternd_max()

    fn test_scatternd_min() raises:
        print("== test_scatternd_min")
        var data = TestTensor[DType.float32, 3](
            DimList(4, 4, 4),
            List[Float32](
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
            ),
        )

        var indices = TestTensor[DType.int64, 2](
            DimList(2, 1), List[Int64](0, 0)
        )

        var updates = TestTensor[DType.float32, 3](
            DimList(2, 4, 4),
            List[Float32](
                5,
                5,
                5,
                5,
                6,
                6,
                6,
                6,
                7,
                7,
                7,
                7,
                8,
                8,
                8,
                8,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                4,
                4,
                4,
                4,
            ),
        )

        var output_ref = TestTensor[DType.float32, 3](
            DimList(4, 4, 4),
            List[Float32](
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                4,
                3,
                2,
                1,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
            ),
        )

        @always_inline
        @parameter
        fn _min[
            ty: DType, width: Int
        ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
            return min(v1, v2)

        test_case[DType.float32, _min](data, indices, updates, output_ref)

    test_scatternd_min()

    fn test_scatternd_multiply() raises:
        print("== test_scatternd_multiply")
        var data = TestTensor[DType.float32, 3](
            DimList(4, 4, 4),
            List[Float32](
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
            ),
        )

        var indices = TestTensor[DType.int64, 2](
            DimList(2, 1), List[Int64](0, 0)
        )

        var updates = TestTensor[DType.float32, 3](
            DimList(2, 4, 4),
            List[Float32](
                5,
                5,
                5,
                5,
                6,
                6,
                6,
                6,
                7,
                7,
                7,
                7,
                8,
                8,
                8,
                8,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                4,
                4,
                4,
                4,
            ),
        )

        var output_ref = TestTensor[DType.float32, 3](
            DimList(4, 4, 4),
            List[Float32](
                5,
                10,
                15,
                20,
                60,
                72,
                84,
                96,
                168,
                147,
                126,
                105,
                128,
                96,
                64,
                32,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
            ),
        )

        @always_inline
        @parameter
        fn _mul[
            ty: DType, width: Int
        ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
            return v1 * v2

        test_case[DType.float32, _mul](data, indices, updates, output_ref)

    test_scatternd_multiply()

    fn test_scatternd_empty_1d() raises:
        print("== test_scatternd_empty_1d")
        # Test 1D scatter_nd with empty updates (identity operation).
        var data = TestTensor[DType.float32, 1](
            DimList(5),
            List[Float32](1.0, 2.0, 3.0, 4.0, 5.0),
        )

        # Empty indices and updates.
        var indices = TestTensor[DType.int64, 2](DimList(0, 1), List[Int64]())
        var updates = TestTensor[DType.float32, 1](DimList(0), List[Float32]())

        # Output should equal input (identity).
        var output_ref = TestTensor[DType.float32, 1](
            DimList(5),
            List[Float32](1.0, 2.0, 3.0, 4.0, 5.0),
        )

        # Create a proper test case function for 1D.
        var output = TestTensor[DType.float32, 1](DimList(5))
        scatter_nd_generator[
            DType.float32, DType.int64, False, reduce_fn=use_update
        ](
            data.to_managed_tensor_slice().to_layout_tensor(),
            indices.to_managed_tensor_slice().to_layout_tensor(),
            updates.to_managed_tensor_slice().to_layout_tensor(),
            output.to_managed_tensor_slice().to_layout_tensor(),
        )
        assert_equal(output, output_ref)

    test_scatternd_empty_1d()

    fn test_scatternd_empty_2d() raises:
        print("== test_scatternd_empty_2d")
        # Test 2D scatter_nd with empty row updates.
        var data = TestTensor[DType.float32, 2](
            DimList(2, 2),
            List[Float32](1.0, 2.0, 3.0, 4.0),
        )

        var indices = TestTensor[DType.int64, 2](DimList(0, 1), List[Int64]())
        var updates = TestTensor[DType.float32, 2](
            DimList(0, 2), List[Float32]()
        )

        var output_ref = TestTensor[DType.float32, 2](
            DimList(2, 2),
            List[Float32](1.0, 2.0, 3.0, 4.0),
        )

        var output = TestTensor[DType.float32, 2](DimList(2, 2))
        scatter_nd_generator[
            DType.float32, DType.int64, False, reduce_fn=use_update
        ](
            data.to_managed_tensor_slice().to_layout_tensor(),
            indices.to_managed_tensor_slice().to_layout_tensor(),
            updates.to_managed_tensor_slice().to_layout_tensor(),
            output.to_managed_tensor_slice().to_layout_tensor(),
        )
        assert_equal(output, output_ref)

    test_scatternd_empty_2d()

    fn test_scatternd_empty_2d_points() raises:
        print("== test_scatternd_empty_2d_points")
        # Test 2D scatter_nd with empty point updates.
        var data = TestTensor[DType.float32, 2](
            DimList(3, 3),
            List[Float32](1, 2, 3, 4, 5, 6, 7, 8, 9),
        )

        var indices = TestTensor[DType.int64, 2](DimList(0, 2), List[Int64]())
        var updates = TestTensor[DType.float32, 1](DimList(0), List[Float32]())

        var output_ref = TestTensor[DType.float32, 2](
            DimList(3, 3),
            List[Float32](1, 2, 3, 4, 5, 6, 7, 8, 9),
        )

        var output = TestTensor[DType.float32, 2](DimList(3, 3))
        scatter_nd_generator[
            DType.float32, DType.int64, False, reduce_fn=use_update
        ](
            data.to_managed_tensor_slice().to_layout_tensor(),
            indices.to_managed_tensor_slice().to_layout_tensor(),
            updates.to_managed_tensor_slice().to_layout_tensor(),
            output.to_managed_tensor_slice().to_layout_tensor(),
        )
        assert_equal(output, output_ref)

    test_scatternd_empty_2d_points()

    fn test_scatternd_empty_3d() raises:
        print("== test_scatternd_empty_3d")
        # Test 3D scatter_nd with empty updates.
        var data = TestTensor[DType.float32, 3](
            DimList(2, 2, 2),
            List[Float32](1, 2, 3, 4, 5, 6, 7, 8),
        )

        var indices = TestTensor[DType.int64, 2](DimList(0, 1), List[Int64]())
        var updates = TestTensor[DType.float32, 3](
            DimList(0, 2, 2), List[Float32]()
        )

        var output_ref = TestTensor[DType.float32, 3](
            DimList(2, 2, 2),
            List[Float32](1, 2, 3, 4, 5, 6, 7, 8),
        )

        test_case[DType.float32](
            data,
            indices,
            updates,
            output_ref,
        )

    test_scatternd_empty_3d()

    fn test_scatternd_single_update() raises:
        print("== test_scatternd_single_update")
        # Test basic functionality with single element update.
        var data = TestTensor[DType.float32, 1](
            DimList(5),
            List[Float32](1.0, 2.0, 3.0, 4.0, 5.0),
        )

        var indices = TestTensor[DType.int64, 2](DimList(1, 1), List[Int64](2))
        var updates = TestTensor[DType.float32, 1](
            DimList(1), List[Float32](99.0)
        )

        var output_ref = TestTensor[DType.float32, 1](
            DimList(5),
            List[Float32](1.0, 2.0, 99.0, 4.0, 5.0),
        )

        var output = TestTensor[DType.float32, 1](DimList(5))
        scatter_nd_generator[
            DType.float32, DType.int64, False, reduce_fn=use_update
        ](
            data.to_managed_tensor_slice().to_layout_tensor(),
            indices.to_managed_tensor_slice().to_layout_tensor(),
            updates.to_managed_tensor_slice().to_layout_tensor(),
            output.to_managed_tensor_slice().to_layout_tensor(),
        )
        assert_equal(output, output_ref)

    test_scatternd_single_update()

    fn test_scatternd_row_updates() raises:
        print("== test_scatternd_row_updates")
        # Test 2D scatter_nd updating entire rows.
        var data = TestTensor[DType.float32, 2](
            DimList(3, 3),
            List[Float32](1, 2, 3, 4, 5, 6, 7, 8, 9),
        )

        var indices = TestTensor[DType.int64, 2](
            DimList(2, 1), List[Int64](0, 2)
        )
        var updates = TestTensor[DType.float32, 2](
            DimList(2, 3), List[Float32](10, 11, 12, 20, 21, 22)
        )

        var output_ref = TestTensor[DType.float32, 2](
            DimList(3, 3),
            List[Float32](10, 11, 12, 4, 5, 6, 20, 21, 22),
        )

        var output = TestTensor[DType.float32, 2](DimList(3, 3))
        scatter_nd_generator[
            DType.float32, DType.int64, False, reduce_fn=use_update
        ](
            data.to_managed_tensor_slice().to_layout_tensor(),
            indices.to_managed_tensor_slice().to_layout_tensor(),
            updates.to_managed_tensor_slice().to_layout_tensor(),
            output.to_managed_tensor_slice().to_layout_tensor(),
        )
        assert_equal(output, output_ref)

    test_scatternd_row_updates()

    fn test_scatternd_point_updates() raises:
        print("== test_scatternd_point_updates")
        # Test 2D scatter_nd updating individual points.
        var data = TestTensor[DType.float32, 2](
            DimList(3, 3),
            List[Float32](1, 2, 3, 4, 5, 6, 7, 8, 9),
        )

        var indices = TestTensor[DType.int64, 2](
            DimList(3, 2), List[Int64](0, 0, 1, 1, 2, 2)
        )
        var updates = TestTensor[DType.float32, 1](
            DimList(3), List[Float32](100, 200, 300)
        )

        var output_ref = TestTensor[DType.float32, 2](
            DimList(3, 3),
            List[Float32](100, 2, 3, 4, 200, 6, 7, 8, 300),
        )

        var output = TestTensor[DType.float32, 2](DimList(3, 3))
        scatter_nd_generator[
            DType.float32, DType.int64, False, reduce_fn=use_update
        ](
            data.to_managed_tensor_slice().to_layout_tensor(),
            indices.to_managed_tensor_slice().to_layout_tensor(),
            updates.to_managed_tensor_slice().to_layout_tensor(),
            output.to_managed_tensor_slice().to_layout_tensor(),
        )
        assert_equal(output, output_ref)

    test_scatternd_point_updates()

    fn test_scatternd_negative_indices() raises:
        print("== test_scatternd_negative_indices")
        # Test negative index wrapping.
        var data = TestTensor[DType.float32, 1](
            DimList(5),
            List[Float32](1.0, 2.0, 3.0, 4.0, 5.0),
        )

        # -1 should wrap to index 4, -3 should wrap to index 2.
        var indices = TestTensor[DType.int64, 2](
            DimList(2, 1), List[Int64](-1, -3)
        )
        var updates = TestTensor[DType.float32, 1](
            DimList(2), List[Float32](100.0, 200.0)
        )

        var output_ref = TestTensor[DType.float32, 1](
            DimList(5),
            List[Float32](1.0, 2.0, 200.0, 4.0, 100.0),
        )

        var output = TestTensor[DType.float32, 1](DimList(5))
        scatter_nd_generator[
            DType.float32, DType.int64, False, reduce_fn=use_update
        ](
            data.to_managed_tensor_slice().to_layout_tensor(),
            indices.to_managed_tensor_slice().to_layout_tensor(),
            updates.to_managed_tensor_slice().to_layout_tensor(),
            output.to_managed_tensor_slice().to_layout_tensor(),
        )
        assert_equal(output, output_ref)

    test_scatternd_negative_indices()

    fn test_scatternd_int32_indices() raises:
        print("== test_scatternd_int32_indices")
        # Test with int32 indices instead of int64.
        var data = TestTensor[DType.float32, 2](
            DimList(3, 3),
            List[Float32](1, 2, 3, 4, 5, 6, 7, 8, 9),
        )

        var indices = TestTensor[DType.int32, 2](
            DimList(2, 2), List[Int32](0, 1, 2, 0)
        )
        var updates = TestTensor[DType.float32, 1](
            DimList(2), List[Float32](100, 200)
        )

        var output_ref = TestTensor[DType.float32, 2](
            DimList(3, 3),
            List[Float32](1, 100, 3, 4, 5, 6, 200, 8, 9),
        )

        var output = TestTensor[DType.float32, 2](DimList(3, 3))
        scatter_nd_generator[
            DType.float32, DType.int32, False, reduce_fn=use_update
        ](
            data.to_managed_tensor_slice().to_layout_tensor(),
            indices.to_managed_tensor_slice().to_layout_tensor(),
            updates.to_managed_tensor_slice().to_layout_tensor(),
            output.to_managed_tensor_slice().to_layout_tensor(),
        )
        assert_equal(output, output_ref)

    test_scatternd_int32_indices()
