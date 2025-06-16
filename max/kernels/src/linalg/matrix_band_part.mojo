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
"""The module implements matrix band part functions."""


from algorithm.functional import elementwise, unswitch
from buffer import NDBuffer
from runtime.asyncrt import DeviceContextPtr

from utils.index import IndexList


@always_inline
fn matrix_band_part[
    type: DType,
    int_type: DType,
    cond_type: DType,
    rank: Int,
    input_0_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing [
        _
    ] -> SIMD[type, width],
    simd_width: Int,
    single_thread_blocking_override: Bool,
    target: StaticString = "cpu",
](
    input_shape: IndexList[rank],
    num_lower: NDBuffer[int_type, 1],
    num_upper: NDBuffer[int_type, 1],
    exclude_buf: NDBuffer[cond_type, 1],
    output: NDBuffer[mut=True, type, rank],
    ctx: DeviceContextPtr,
) raises:
    var lower_diagonal_index = Int(num_lower[0])
    var upper_diagonal_index = Int(num_upper[0])

    @__copy_capture(
        input_shape, lower_diagonal_index, upper_diagonal_index, output
    )
    @parameter
    fn dispatch[exclude: Bool]() raises:
        _matrix_band_part_impl[
            type,
            int_type,
            cond_type,
            rank,
            input_0_fn,
            simd_width,
            single_thread_blocking_override,
            exclude=exclude,
            target=target,
        ](input_shape, lower_diagonal_index, upper_diagonal_index, output, ctx)

    unswitch[dispatch](exclude_buf[0] != 0)


@always_inline
fn _matrix_band_part_impl[
    type: DType,
    int_type: DType,
    cond_type: DType,
    rank: Int,
    input_0_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing [
        _
    ] -> SIMD[type, width],
    simd_width: Int,
    single_thread_blocking_override: Bool,
    exclude: Bool,
    target: StaticString = "cpu",
](
    input_shape: IndexList[rank],
    lower_diagonal_index: Int,
    upper_diagonal_index: Int,
    output: NDBuffer[mut=True, type, rank],
    ctx: DeviceContextPtr,
) raises:
    constrained[rank >= 2, "Matrix band only supports rank >=2"]()

    @__copy_capture(lower_diagonal_index, upper_diagonal_index, output)
    @parameter
    @always_inline
    fn func[simd_width: Int, inner_rank: Int](index: IndexList[inner_rank]):
        var idx = rebind[IndexList[rank]](index)

        var row = idx[rank - 2]
        var col = idx[rank - 1]

        var in_band = (
            lower_diagonal_index < 0 or (row - col) <= lower_diagonal_index
        ) and (upper_diagonal_index < 0 or (col - row) <= upper_diagonal_index)

        @parameter
        if exclude:
            in_band = not in_band

        if in_band:
            output[idx] = rebind[Scalar[type]](
                input_0_fn[simd_width, rank](idx)
            )
        else:
            output[idx] = 0

    elementwise[
        func,
        simd_width=1,
        use_blocking_impl=single_thread_blocking_override,
        target=target,
    ](input_shape, context=ctx)
