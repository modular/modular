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
from layout import IntTuple, Layout, LayoutTensor
from layout.math import max, sum
from std.testing import assert_equal


def test_strided_mixed_dtype() raises:
    var input_storage = Array[Float16, 12](fill=-99)
    var sum_storage = Array[Float32, 6](fill=-77)
    var max_storage = Array[Float32, 6](fill=-77)
    comptime input_layout = Layout(IntTuple(2, 3), IntTuple(1, 4))
    comptime output_layout = Layout(IntTuple(3), IntTuple(2))
    var inp = LayoutTensor[.float16, input_layout](input_storage)
    var summed = LayoutTensor[.float32, output_layout](sum_storage)
    var maxima = LayoutTensor[.float32, output_layout](max_storage)
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
    var inp = LayoutTensor[.float32, Layout.row_major(2, 12)](
        storage
    ).vectorize[1, 4]()
    # Legacy allocated reductions retain scalar stride 1 for SIMD elements;
    # explicit vectorized outputs keep independent cells four scalars apart.
    var rows_sum = (
        LayoutTensor[mut=True, .float32, Layout.row_major(8), MutAnyOrigin]
        .stack_allocation[stack_alignment=16]()
        .vectorize[4]()
    )
    var rows_max = (
        LayoutTensor[mut=True, .float32, Layout.row_major(8), MutAnyOrigin]
        .stack_allocation[stack_alignment=16]()
        .vectorize[4]()
    )
    var cols_sum = (
        LayoutTensor[mut=True, .float32, Layout.row_major(12), MutAnyOrigin]
        .stack_allocation[stack_alignment=16]()
        .vectorize[4]()
    )
    var cols_max = (
        LayoutTensor[mut=True, .float32, Layout.row_major(12), MutAnyOrigin]
        .stack_allocation[stack_alignment=16]()
        .vectorize[4]()
    )
    sum[1](inp, rows_sum)
    sum[0](inp, cols_sum)
    max[1](inp, rows_max)
    max[0](inp, cols_max)
    for i in range(2):
        for lane in range(4):
            assert_equal(rows_sum[i][lane], Float32(36 * i + 3 * lane - 78))
            assert_equal(rows_max[i][lane], Float32(12 * i + lane - 22))
    for j in range(3):
        for lane in range(4):
            assert_equal(cols_sum[j][lane], Float32(8 * j + 2 * lane - 48))
            assert_equal(cols_max[j][lane], Float32(4 * j + lane - 18))


def main() raises:
    test_strided_mixed_dtype()
    test_vector_elements()
