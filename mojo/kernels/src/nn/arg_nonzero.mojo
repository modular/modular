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

from collections.string import StaticString

from algorithm import sync_parallelize
from algorithm.functional import _get_start_indices_of_nth_subvolume
from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from register import register_internal
from runtime.tracing import Trace, TraceLevel

from utils.index import IndexList

# ===-----------------------------------------------------------------------===#
# arg_nonzero
# ===-----------------------------------------------------------------------===#


@register_internal("mo.arg_nonzero")
@always_inline
fn arg_nonzero[
    type: DType,
    output_type: DType,
    rank: Int,
](
    input_buffer: NDBuffer[type, rank],
    output_buffer: NDBuffer[mut=True, output_type, 2],
):
    """Gather the indices of all non-zero elements in input buffer storing
    the indices in the output_buffer.

    Parameters:
        type: The element type.
        output_type: The integer type to store the indices in.
        rank: The rank of the tensor.

    Args:
        input_buffer: The tensor to count the non-zeros in.
        output_buffer: The indices of all non-zero elements.
    """

    with Trace[TraceLevel.OP, target = StaticString("cpu")]("arg_nonzero"):
        var numel = input_buffer.get_shape().flattened_length()
        if numel == 0:
            return

        var j: Int = 0
        for i in range(numel):
            var indices = _get_start_indices_of_nth_subvolume[0](
                i, input_buffer.get_shape()
            )
            if input_buffer[indices]:
                var out_indices = IndexList[2]()
                out_indices[0] = j
                j += 1

                # Write each of the output values to the output buffer.
                @parameter
                for k in range(rank):
                    out_indices[1] = k
                    output_buffer[out_indices] = indices[k]


# Where has the shape 2D shape [NumNonZeros, InputRank]
@always_inline
fn arg_nonzero_shape[
    type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
](input_buffer: NDBuffer[type, rank],) -> IndexList[2]:
    """Return [NumNonZeros, InputRank] where NumNonZeros are the number of
    non-zero elements in the input.

    Parameters:
        type: The element type.
        rank: The rank.
        single_thread_blocking_override: This op can block.

    Args:
        input_buffer: The tensor to count the non-zeros in.

    Returns:
        Shape of the arg_nonzero kernel for this input [NumNonZeros, InputRank].
    """

    var shape = IndexList[2]()
    shape[1] = rank

    var numel = input_buffer.get_shape().flattened_length()

    var j: Int = 0
    for i in range(numel):
        var indices = _get_start_indices_of_nth_subvolume[0](
            i, input_buffer.get_shape()
        )
        if input_buffer[indices]:
            j += 1

    shape[0] = j
    return shape
