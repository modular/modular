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

from layout import TileTensor, row_major
from nn.gather_scatter import scatter_nd_generator
from std.testing import assert_equal


@always_inline
@parameter
def use_update[
    dtype: DType, width: Int, //
](input_val: SIMD[dtype, width], update_val: SIMD[dtype, width]) -> SIMD[
    dtype, width
]:
    return update_val


def main() raises:
    def test_scatternd() raises:
        print("== test_scatternd")
        # data: 4x4x4 = 64 elements
        var data_ptr = alloc[Float32](64)
        var data_vals: InlineArray[Float32, 64] = [
            Float32(1),
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
        ]
        for i in range(64):
            data_ptr[i] = data_vals[i]

        var data = TileTensor(data_ptr, row_major[4, 4, 4]())

        # indices: 2x1 = 2 elements
        var indices_ptr = alloc[Int64](2)
        indices_ptr[0] = 0
        indices_ptr[1] = 2

        var indices = TileTensor(indices_ptr, row_major[2, 1]())

        # updates: 2x4x4 = 32 elements
        var updates_ptr = alloc[Float32](32)
        var updates_vals: InlineArray[Float32, 32] = [
            Float32(5),
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
        ]
        for i in range(32):
            updates_ptr[i] = updates_vals[i]

        var updates = TileTensor(updates_ptr, row_major[2, 4, 4]())

        # output: 4x4x4 = 64 elements
        var output_ptr = alloc[Float32](64)
        var output = TileTensor(output_ptr, row_major[4, 4, 4]())

        # expected output
        var expected: InlineArray[Float32, 64] = [
            Float32(5),
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
        ]

        scatter_nd_generator[reduce_fn=use_update](
            data, indices, updates, output
        )

        for i in range(64):
            assert_equal(output_ptr[i], expected[i])

        data_ptr.free()
        indices_ptr.free()
        updates_ptr.free()
        output_ptr.free()

    test_scatternd()

    def test_scatternd_add() raises:
        print("== test_scatternd_add")
        # data: 4x4x4 = 64 elements
        var data_ptr = alloc[Float32](64)
        var data_vals: InlineArray[Float32, 64] = [
            Float32(1),
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
        ]
        for i in range(64):
            data_ptr[i] = data_vals[i]

        var data = TileTensor(data_ptr, row_major[4, 4, 4]())

        # indices: 2x1 = 2 elements (both pointing to index 0)
        var indices_ptr = alloc[Int64](2)
        indices_ptr[0] = 0
        indices_ptr[1] = 0

        var indices = TileTensor(indices_ptr, row_major[2, 1]())

        # updates: 2x4x4 = 32 elements
        var updates_ptr = alloc[Float32](32)
        var updates_vals: InlineArray[Float32, 32] = [
            Float32(5),
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
        ]
        for i in range(32):
            updates_ptr[i] = updates_vals[i]

        var updates = TileTensor(updates_ptr, row_major[2, 4, 4]())

        # output: 4x4x4 = 64 elements
        var output_ptr = alloc[Float32](64)
        var output = TileTensor(output_ptr, row_major[4, 4, 4]())

        # expected output (add reduction)
        var expected: InlineArray[Float32, 64] = [
            Float32(7),
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
        ]

        @always_inline
        @parameter
        def _add[
            ty: DType, width: Int
        ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
            return v1 + v2

        scatter_nd_generator[reduce_fn=_add](data, indices, updates, output)

        for i in range(64):
            assert_equal(output_ptr[i], expected[i])

        data_ptr.free()
        indices_ptr.free()
        updates_ptr.free()
        output_ptr.free()

    test_scatternd_add()

    def test_scatternd_max() raises:
        print("== test_scatternd_max")
        # data: 4x4x4 = 64 elements
        var data_ptr = alloc[Float32](64)
        var data_vals: InlineArray[Float32, 64] = [
            Float32(1),
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
        ]
        for i in range(64):
            data_ptr[i] = data_vals[i]

        var data = TileTensor(data_ptr, row_major[4, 4, 4]())

        # indices: 2x1 = 2 elements (both pointing to index 0)
        var indices_ptr = alloc[Int64](2)
        indices_ptr[0] = 0
        indices_ptr[1] = 0

        var indices = TileTensor(indices_ptr, row_major[2, 1]())

        # updates: 2x4x4 = 32 elements
        var updates_ptr = alloc[Float32](32)
        var updates_vals: InlineArray[Float32, 32] = [
            Float32(5),
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
        ]
        for i in range(32):
            updates_ptr[i] = updates_vals[i]

        var updates = TileTensor(updates_ptr, row_major[2, 4, 4]())

        # output: 4x4x4 = 64 elements
        var output_ptr = alloc[Float32](64)
        var output = TileTensor(output_ptr, row_major[4, 4, 4]())

        # expected output (max reduction)
        var expected: InlineArray[Float32, 64] = [
            Float32(5),
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
        ]

        @always_inline
        @parameter
        def _max[
            ty: DType, width: Int
        ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
            return max(v1, v2)

        scatter_nd_generator[reduce_fn=_max](data, indices, updates, output)

        for i in range(64):
            assert_equal(output_ptr[i], expected[i])

        data_ptr.free()
        indices_ptr.free()
        updates_ptr.free()
        output_ptr.free()

    test_scatternd_max()

    def test_scatternd_min() raises:
        print("== test_scatternd_min")
        # data: 4x4x4 = 64 elements
        var data_ptr = alloc[Float32](64)
        var data_vals: InlineArray[Float32, 64] = [
            Float32(1),
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
        ]
        for i in range(64):
            data_ptr[i] = data_vals[i]

        var data = TileTensor(data_ptr, row_major[4, 4, 4]())

        # indices: 2x1 = 2 elements (both pointing to index 0)
        var indices_ptr = alloc[Int64](2)
        indices_ptr[0] = 0
        indices_ptr[1] = 0

        var indices = TileTensor(indices_ptr, row_major[2, 1]())

        # updates: 2x4x4 = 32 elements
        var updates_ptr = alloc[Float32](32)
        var updates_vals: InlineArray[Float32, 32] = [
            Float32(5),
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
        ]
        for i in range(32):
            updates_ptr[i] = updates_vals[i]

        var updates = TileTensor(updates_ptr, row_major[2, 4, 4]())

        # output: 4x4x4 = 64 elements
        var output_ptr = alloc[Float32](64)
        var output = TileTensor(output_ptr, row_major[4, 4, 4]())

        # expected output (min reduction)
        var expected: InlineArray[Float32, 64] = [
            Float32(1),
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
        ]

        @always_inline
        @parameter
        def _min[
            ty: DType, width: Int
        ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
            return min(v1, v2)

        scatter_nd_generator[reduce_fn=_min](data, indices, updates, output)

        for i in range(64):
            assert_equal(output_ptr[i], expected[i])

        data_ptr.free()
        indices_ptr.free()
        updates_ptr.free()
        output_ptr.free()

    test_scatternd_min()

    def test_scatternd_multiply() raises:
        print("== test_scatternd_multiply")
        # data: 4x4x4 = 64 elements
        var data_ptr = alloc[Float32](64)
        var data_vals: InlineArray[Float32, 64] = [
            Float32(1),
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
        ]
        for i in range(64):
            data_ptr[i] = data_vals[i]

        var data = TileTensor(data_ptr, row_major[4, 4, 4]())

        # indices: 2x1 = 2 elements (both pointing to index 0)
        var indices_ptr = alloc[Int64](2)
        indices_ptr[0] = 0
        indices_ptr[1] = 0

        var indices = TileTensor(indices_ptr, row_major[2, 1]())

        # updates: 2x4x4 = 32 elements
        var updates_ptr = alloc[Float32](32)
        var updates_vals: InlineArray[Float32, 32] = [
            Float32(5),
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
        ]
        for i in range(32):
            updates_ptr[i] = updates_vals[i]

        var updates = TileTensor(updates_ptr, row_major[2, 4, 4]())

        # output: 4x4x4 = 64 elements
        var output_ptr = alloc[Float32](64)
        var output = TileTensor(output_ptr, row_major[4, 4, 4]())

        # expected output (multiply reduction)
        var expected: InlineArray[Float32, 64] = [
            Float32(5),
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
        ]

        @always_inline
        @parameter
        def _mul[
            ty: DType, width: Int
        ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
            return v1 * v2

        scatter_nd_generator[reduce_fn=_mul](data, indices, updates, output)

        for i in range(64):
            assert_equal(output_ptr[i], expected[i])

        data_ptr.free()
        indices_ptr.free()
        updates_ptr.free()
        output_ptr.free()

    test_scatternd_multiply()
