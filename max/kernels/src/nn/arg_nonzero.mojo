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


from algorithm.functional import _get_start_indices_of_nth_subvolume
from layout import LayoutTensor, RuntimeTuple, IntTuple
from layout.int_tuple import fill_like
from runtime.tracing import Trace, TraceLevel

from utils.index import IndexList

# ===-----------------------------------------------------------------------===#
# arg_nonzero
# ===-----------------------------------------------------------------------===#


@always_inline
fn arg_nonzero[
    dtype: DType,
    output_type: DType,
](
    input_buffer: LayoutTensor[dtype, **_],
    output_buffer: LayoutTensor[mut=True, output_type, **_],
) raises:
    """Gather the indices of all non-zero elements in input buffer storing
    the indices in the output_buffer.

    Parameters:
        dtype: The element dtype.
        output_type: The integer dtype to store the indices in.

    Args:
        input_buffer: The tensor to count the non-zeros in.
        output_buffer: The indices of all non-zero elements.
    """
    constrained[output_buffer.rank == 2, "output_buffer must be of rank 2"]()

    with Trace[TraceLevel.OP, target = StaticString("cpu")]("arg_nonzero"):
        var numel = input_buffer.size()
        if numel == 0:
            return

        var j: Int = 0
        for i in range(numel):
            var indices = _get_start_indices_of_nth_subvolume[0](
                i, input_buffer.runtime_layout.shape.value
            )
            var offset = input_buffer.runtime_layout(
                RuntimeTuple[
                    fill_like(input_buffer.layout.shape, 1),
                    element_type = indices.element_type,
                ](indices)
            )
            if input_buffer.ptr.load(offset) != 0:
                var out_indices = IndexList[2]()
                out_indices[0] = j
                j += 1

                # Get the coordinates for the current index
                var coords = input_buffer.runtime_layout.idx2crd(
                    RuntimeTuple[IntTuple(1)](Int(offset))
                )

                # Write each coordinate to the output buffer
                @parameter
                for k in range(input_buffer.rank):
                    out_indices[1] = k
                    output_buffer[out_indices[0], out_indices[1]] = Int(
                        coords[k]
                    )


# Where has the shape 2D shape [NumNonZeros, InputRank]
@always_inline
fn arg_nonzero_shape[
    dtype: DType,
    single_thread_blocking_override: Bool,
](input_buffer: LayoutTensor[dtype, **_]) -> IndexList[2]:
    """Return [NumNonZeros, InputRank] where NumNonZeros are the number of
    non-zero elements in the input.

    Parameters:
        dtype: The element dtype.
        single_thread_blocking_override: This op can block.

    Args:
        input_buffer: The tensor to count the non-zeros in.

    Returns:
        Shape of the arg_nonzero kernel for this input [NumNonZeros, InputRank].
    """

    var shape = IndexList[2]()
    shape[1] = input_buffer.rank

    var numel = input_buffer.size()

    var j: Int = 0
    for i in range(numel):
        var offset = input_buffer.runtime_layout(i)
        if input_buffer.ptr.load(offset) != 0:
            j += 1

    shape[0] = j
    return shape
