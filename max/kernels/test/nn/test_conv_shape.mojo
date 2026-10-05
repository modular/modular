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

from layout import Coord, TileTensor, row_major
from nn.conv.conv import conv_shape
from std.testing import assert_equal
from std.utils.index import IndexList


def check_shape[
    rank: Int, filter_rank: Int
](
    input_shape: IndexList[rank],
    filter_shape: IndexList[filter_rank],
    var strides: List[Int64],
    var dilations: List[Int64],
    var paddings: List[Int64],
    groups: Int,
    expected: IndexList[rank],
    expected_error: String = "",
) raises:
    var input = List(length=input_shape.flattened_length(), fill=Float32(0))
    var filter = List(length=filter_shape.flattened_length(), fill=Float32(0))
    var input_tt = TileTensor(Span(input), row_major(len(input))).reshape(
        Coord(input_shape)
    )
    var filter_tt = TileTensor(Span(filter), row_major(len(filter))).reshape(
        Coord(filter_shape)
    )
    var strides_tt = TileTensor(Span(strides), row_major(len(strides)))
    var dilations_tt = TileTensor(Span(dilations), row_major(len(dilations)))
    var paddings_tt = TileTensor(Span(paddings), row_major(len(paddings)))
    comptime assert input_tt.rank == input_tt.flat_rank == rank
    var result = IndexList[rank]()
    var error_message = String()
    try:
        result = rebind[IndexList[rank]](
            conv_shape(
                input_tt.as_imm(),
                filter_tt.as_imm(),
                strides_tt.as_imm(),
                dilations_tt.as_imm(),
                paddings_tt.as_imm(),
                Int64(groups),
            )
        )
    except e:
        error_message = String(e)
    assert_equal(error_message, expected_error)
    if expected_error == "":
        comptime for i in range(rank):
            assert_equal(result[i], expected[i])


def main() raises:
    # Distinct padding ends, strides, and dilations exercise each spatial axis.
    check_shape(
        IndexList[3](2, 11, 4),
        IndexList[3](3, 2, 6),
        [Int64(2)],
        [Int64(2)],
        [Int64(1), Int64(2)],
        2,
        IndexList[3](2, 5, 6),
    )
    check_shape(
        IndexList[4](2, 7, 9, 4),
        IndexList[4](3, 2, 2, 6),
        [Int64(2), Int64(3)],
        [Int64(1), Int64(2)],
        [Int64(1), Int64(0), Int64(2), Int64(1)],
        2,
        IndexList[4](2, 3, 4, 6),
    )
    check_shape(
        IndexList[5](1, 5, 6, 7, 2),
        IndexList[5](2, 3, 2, 2, 4),
        [Int64(1), Int64(2), Int64(3)],
        [Int64(2), Int64(1), Int64(1)],
        [Int64(0), Int64(0), Int64(1), Int64(1), Int64(0), Int64(1)],
        1,
        IndexList[5](1, 3, 3, 3, 4),
    )
    # Empty spatial dimensions stay empty even when the filter is larger.
    check_shape(
        IndexList[4](1, 0, 9, 2),
        IndexList[4](3, 3, 2, 4),
        [Int64(2), Int64(2)],
        [Int64(1), Int64(1)],
        [Int64(0), Int64(0), Int64(0), Int64(0)],
        1,
        IndexList[4](1, 0, 4, 4),
    )
    check_shape(
        IndexList[2](2, 4),
        IndexList[2](2, 4),
        [],
        [],
        [],
        1,
        IndexList[2](),
        "[convolution] requires (input_rank >= 3)",
    )
    check_shape(
        IndexList[3](1, 5, 2),
        IndexList[4](1, 3, 2, 4),
        [Int64(1)],
        [Int64(1)],
        [Int64(0), Int64(0)],
        1,
        IndexList[3](),
        "[convolution] requires (input_rank == filter_rank)",
    )
    check_shape(
        IndexList[3](1, 5, 2),
        IndexList[3](3, 2, 4),
        [Int64(1), Int64(1)],
        [Int64(1)],
        [Int64(0), Int64(0)],
        1,
        IndexList[3](),
        (
            "[convolution] requires (len(strides) == len(dilations) =="
            " input_rank - 2)"
        ),
    )
    check_shape(
        IndexList[3](1, 5, 2),
        IndexList[3](3, 2, 4),
        [Int64(1)],
        [Int64(1)],
        [Int64(0)],
        1,
        IndexList[3](),
        "[convolution] requires (len(paddings) == 2 * (input rank - 2))",
    )
    check_shape(
        IndexList[3](1, 5, 3),
        IndexList[3](3, 2, 4),
        [Int64(1)],
        [Int64(1)],
        [Int64(0), Int64(0)],
        1,
        IndexList[3](),
        (
            "[convolution] requires (input_channels == num_groups *"
            " filter_channels)"
        ),
    )
    check_shape(
        IndexList[3](1, 5, 4),
        IndexList[3](3, 2, 3),
        [Int64(1)],
        [Int64(1)],
        [Int64(0), Int64(0)],
        2,
        IndexList[3](),
        "[convolution] output_channels must be divisible by num_groups",
    )
    check_shape(
        IndexList[3](1, 2, 2),
        IndexList[3](3, 2, 4),
        [Int64(1)],
        [Int64(1)],
        [Int64(0), Int64(0)],
        1,
        IndexList[3](),
        "[convolution] output spatial dim must be positive",
    )
