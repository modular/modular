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
from layout import Coord, TileTensor, coord, row_major, stack_allocation
from layout.tile_layout import Layout
from layout.math import max, sum
from std.testing import assert_equal


def test_strided_mixed_dtype() raises:
    var input_storage = Array[Float16, 12](fill=-99)
    var sum_storage = Array[Float32, 6](fill=-77)
    var max_storage = Array[Float32, 6](fill=-77)
    comptime input_layout = Layout(coord[2, 3], coord[1, 4])
    comptime output_layout = Layout(coord[3], coord[2])
    var inp = TileTensor(input_storage, input_layout)
    var summed = TileTensor(sum_storage, output_layout)
    var maxima = TileTensor(max_storage, output_layout)
    for i in range(2):
        for j in range(3):
            inp[i, j] = Float16(-10 + i * 3 + j)
    sum[0](inp, summed)
    max[0](inp, maxima)
    for j in range(3):
        assert_equal(sum_storage[2 * j], Float32(-17 + 2 * j))
        assert_equal(max_storage[2 * j], Float32(-7 + j))
        assert_equal(sum_storage[2 * j + 1], Float32(-77))
        assert_equal(max_storage[2 * j + 1], Float32(-77))
    for i in [2, 3, 6, 7, 10, 11]:
        assert_equal(input_storage[i], Float16(-99))
    var row_sum = sum[1](inp)
    var row_max = max[1](inp)
    for i in range(2):
        assert_equal(row_sum[i], Float16(-27 + 9 * i))
        assert_equal(row_max[i], Float16(-8 + 3 * i))


def test_vector_elements() raises:
    var storage = Array[Float32, 24](fill=0)
    for i in range(24):
        storage[i] = Float32(i - 30)
    var inp = TileTensor(storage, row_major[2, 12]()).vectorize[1, 4]()
    var rows_sum = stack_allocation[.float32, alignment=16](
        row_major[8]()
    ).vectorize[4]()
    var rows_max = stack_allocation[.float32, alignment=16](
        row_major[8]()
    ).vectorize[4]()
    var cols_sum = stack_allocation[.float32, alignment=16](
        row_major[12]()
    ).vectorize[4]()
    var cols_max = stack_allocation[.float32, alignment=16](
        row_major[12]()
    ).vectorize[4]()
    sum[1](inp, rows_sum)
    sum[0](inp, cols_sum)
    max[1](inp, rows_max)
    max[0](inp, cols_max)
    var allocated_rows_sum = sum[1](inp)
    var allocated_rows_max = max[1](inp)
    var allocated_cols_sum = sum[0](inp)
    var allocated_cols_max = max[0](inp)
    for i in range(2):
        assert_equal(allocated_rows_sum[i], rows_sum[i])
        assert_equal(allocated_rows_max[i], rows_max[i])
        for lane in range(4):
            assert_equal(rows_sum[i][lane], Float32(36 * i + 3 * lane - 78))
            assert_equal(rows_max[i][lane], Float32(12 * i + lane - 22))
    for j in range(3):
        assert_equal(allocated_cols_sum[j], cols_sum[j])
        assert_equal(allocated_cols_max[j], cols_max[j])
        for lane in range(4):
            assert_equal(cols_sum[j][lane], Float32(8 * j + 2 * lane - 48))
            assert_equal(cols_max[j][lane], Float32(4 * j + lane - 18))


def test_nested_outer_dimensions() raises:
    var storage = Array[Float32, 16](
        fill_with=lambda (i: Int) -> Float32: Float32(i)
    )
    var inp = TileTensor(
        storage,
        Layout(
            Coord(coord[2, 2], coord[2, 2]),
            Coord(coord[8, 1], coord[4, 2]),
        ),
    )
    var rows_sum = sum[1](inp)
    var rows_max = max[1](inp)
    var cols_sum = sum[0](inp)
    var cols_max = max[0](inp)
    var expected_rows_sum = SIMD[.float32, 4](12, 44, 16, 48)
    var expected_rows_max = SIMD[.float32, 4](6, 14, 7, 15)
    var expected_cols_sum = SIMD[.float32, 4](18, 34, 26, 42)
    var expected_cols_max = SIMD[.float32, 4](9, 13, 11, 15)
    for i in range(4):
        assert_equal(rows_sum[i], expected_rows_sum[i])
        assert_equal(rows_max[i], expected_rows_max[i])
        assert_equal(cols_sum[i], expected_cols_sum[i])
        assert_equal(cols_max[i], expected_cols_max[i])


def main() raises:
    test_strided_mixed_dtype()
    test_vector_elements()
    test_nested_outer_dimensions()
