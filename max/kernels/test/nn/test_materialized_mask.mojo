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

from std.testing import assert_equal
from std.utils import IndexList

from layout import ImmTileTensor, TensorLayout, TileTensor, row_major
from nn.attention.mha_mask import MASK_VALUE, MaterializedMask


def check_mask[rank: Int, columns: Int, explicit_start: Bool]() raises:
    comptime heads = 2 if rank == 4 else 1
    comptime rows = 3
    var data = List(length=2 * heads * rows * columns, fill=Float32(0))
    for i in range(len(data)):
        data[i] = Float32(i - 20)

    comptime if rank == 3:
        var tensor = TileTensor(Span(data), row_major[2, rows, columns]())
        check_values[explicit_start](tensor.as_imm(), data)
    else:
        var tensor = TileTensor(
            Span(data), row_major[2, heads, rows, columns]()
        )
        check_values[explicit_start](tensor.as_imm(), data)


def check_values[
    LayoutType: TensorLayout, origin: ImmOrigin, //, explicit_start: Bool
](
    tensor: ImmTileTensor[.float32, LayoutType, origin], data: List[Float32]
) raises:
    comptime rank = tensor.rank
    comptime columns = tensor.static_shape[rank - 1]
    comptime rows = tensor.static_shape[rank - 2]
    comptime heads = tensor.static_shape[1] if rank == 4 else 1
    var starts = [UInt32(7), UInt32(11)]
    var start_tensor = (
        TileTensor(Span(starts), row_major(Int64(2)))
        .as_imm()
        .as_unsafe_any_origin()
    )

    var mask = MaterializedMask(tensor)
    comptime if explicit_start:
        mask = MaterializedMask(tensor, start_tensor)

    for batch in range(2):
        var start = Int(starts[batch]) if explicit_start else columns - rows
        assert_equal(mask.get_start_pos(batch), start)
        for head in range(2):
            for row in range(rows + 1):
                for col in range(0, columns + 2, 2):
                    var actual = mask.mask(
                        IndexList[4](batch, head, row + start, col),
                        SIMD[.float32, 2](0.25),
                    )
                    comptime for lane in range(2):
                        var expected = Float32(MASK_VALUE)
                        if row < rows and col + lane < columns:
                            var stored_head = head if rank == 4 else 0
                            expected = data[
                                ((batch * heads + stored_head) * rows + row)
                                * columns
                                + col
                                + lane
                            ]
                        assert_equal(actual[lane], expected + 0.25)


def main() raises:
    check_mask[3, 5, False]()
    check_mask[3, 6, True]()
    check_mask[4, 5, True]()
    check_mask[4, 6, False]()
