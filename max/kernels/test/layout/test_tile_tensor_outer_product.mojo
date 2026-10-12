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
from layout import Coord, Idx, TileTensor, col_major, row_major
from layout.tile_layout import Layout as TileLayout
from layout.math import outer_product_acc
from std.testing import assert_equal


def test_strided_mixed_dtype() raises:
    var lhs_storage = Array[Float16, 3](fill=-100)
    var rhs_storage = Array[Float32, 7](fill=-200)
    var result_storage = Array[Float32, 6](fill=0)
    var lhs = TileTensor(lhs_storage, TileLayout(Coord(Idx[2]), Coord(Idx[2])))
    var rhs = TileTensor(rhs_storage, TileLayout(Coord(Idx[3]), Coord(Idx[3])))
    var result = TileTensor(result_storage, col_major[2, 3]())
    for i in range(2):
        lhs[i] = Float16(i + 2)
    for j in range(3):
        rhs[j] = Float32(j + 4)
    for i in range(2):
        for j in range(3):
            result[i, j] = Float32(10 * i + j + 1)

    outer_product_acc(result, lhs.as_imm(), rhs.as_imm())
    outer_product_acc(result, lhs.as_imm(), rhs.as_imm())

    for i in range(2):
        for j in range(3):
            assert_equal(
                result_storage[i + 2 * j],
                Float32(10 * i + j + 1 + 2 * (i + 2) * (j + 4)),
            )
    assert_equal(lhs_storage[1], Float16(-100))
    for j in [1, 2, 4, 5]:
        assert_equal(rhs_storage[j], Float32(-200))


def test_vector_elements() raises:
    var lhs_storage = Array[Float32, 8](fill=0)
    var rhs_storage = Array[Float32, 12](fill=0)
    var result_storage = Array[Float32, 24](fill=0)
    for i in range(8):
        lhs_storage[i] = Float32(i + 1)
    for i in range(12):
        rhs_storage[i] = Float32(2 * i + 1)
    for i in range(24):
        result_storage[i] = Float32(i - 5)
    var lhs = TileTensor(lhs_storage, row_major[8]()).vectorize[4]()
    var rhs = TileTensor(rhs_storage, row_major[12]()).vectorize[4]()
    var result = TileTensor(result_storage, row_major[2, 12]()).vectorize[
        1, 4
    ]()

    outer_product_acc(result, lhs.as_imm(), rhs.as_imm())

    for i in range(2):
        for j in range(3):
            for lane in range(4):
                var offset = (i * 3 + j) * 4 + lane
                assert_equal(
                    result_storage[offset],
                    Float32(
                        offset
                        - 5
                        + (i * 4 + lane + 1) * (2 * (j * 4 + lane) + 1)
                    ),
                )


def main() raises:
    test_strided_mixed_dtype()
    test_vector_elements()
