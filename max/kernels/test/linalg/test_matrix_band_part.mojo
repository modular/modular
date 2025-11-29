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

from layout import Layout, LayoutTensor
from layout.int_tuple import to_index_list
from buffer.dimlist import DimList
from linalg.matrix_band_part import matrix_band_part as _matrix_band_part
from runtime.asyncrt import DeviceContextPtr
from testing import assert_equal

from utils import IndexList


def matrix_band_part[
    output_layout: Layout,
    dtype: DType,
](
    input: LayoutTensor[dtype, output_layout, ImmutAnyOrigin],
    output: LayoutTensor[dtype, output_layout, MutAnyOrigin],
    num_lower: Int,
    num_upper: Int,
    exclude: Bool,
):
    comptime int_type = DType.int
    comptime cond_type = DType.bool

    var num_lower_buf = LayoutTensor[
        int_type, Layout.row_major(1), MutAnyOrigin
    ].stack_allocation()
    var num_upper_buf = LayoutTensor[
        int_type, Layout.row_major(1), MutAnyOrigin
    ].stack_allocation()
    var exclude_buf = LayoutTensor[
        cond_type, Layout.row_major(1), MutAnyOrigin
    ].stack_allocation()

    num_lower_buf[0] = num_lower
    num_upper_buf[0] = num_upper
    exclude_buf[0] = exclude
    comptime rank = input.rank
    var input_shape: IndexList[rank] = to_index_list[rank](input.layout.shape)

    @parameter
    fn input_fn[
        width: Int,
        _rank: Int,
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        return input.load[width=width](rebind[IndexList[rank]](coords))

    _matrix_band_part[
        dtype,
        int_type,
        cond_type,
        rank,
        input_fn,
        simd_width=1,
        single_thread_blocking_override=True,
    ](
        input_shape,
        num_lower_buf.get_immutable(),
        num_upper_buf.get_immutable(),
        exclude_buf.get_immutable(),
        output,
        DeviceContextPtr(),
    )


def test_matrix_band_part():
    comptime layout = Layout.row_major(3, 3)
    comptime dtype = DType.float32

    var input = LayoutTensor[dtype, layout, MutAnyOrigin].stack_allocation()
    var output = LayoutTensor[dtype, layout, MutAnyOrigin].stack_allocation()

    input[0, 0] = 1
    input[0, 1] = 2
    input[0, 2] = 3
    input[1, 0] = 4
    input[1, 1] = 5
    input[1, 2] = 6
    input[2, 0] = 7
    input[2, 1] = 8
    input[2, 2] = 9

    matrix_band_part(
        input.get_immutable(),
        output,
        num_lower=0,
        num_upper=-1,
        exclude=False,
    )

    assert_equal(output[0, 0], 1)
    assert_equal(output[0, 1], 2)
    assert_equal(output[0, 2], 3)
    assert_equal(output[1, 0], 0)
    assert_equal(output[1, 1], 5)
    assert_equal(output[1, 2], 6)
    assert_equal(output[2, 0], 0)
    assert_equal(output[2, 1], 0)
    assert_equal(output[2, 2], 9)

    matrix_band_part(
        input.get_immutable(),
        output,
        num_lower=0,
        num_upper=-1,
        exclude=True,
    )

    assert_equal(output[0, 0], 0)
    assert_equal(output[0, 1], 0)
    assert_equal(output[0, 2], 0)
    assert_equal(output[1, 0], 4)
    assert_equal(output[1, 1], 0)
    assert_equal(output[1, 2], 0)
    assert_equal(output[2, 0], 7)
    assert_equal(output[2, 1], 8)
    assert_equal(output[2, 2], 0)


def main():
    test_matrix_band_part()
