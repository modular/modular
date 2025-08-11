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


from math import ceildiv, iota
from sys.info import simdwidthof

from algorithm import elementwise
from bit import next_power_of_two
from gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx
from gpu.host import DeviceContext
from gpu.host import get_gpu_target
from gpu.host.info import is_cpu
from layout import UNKNOWN_VALUE, Layout, LayoutTensor, RuntimeLayout
from runtime.tracing import Trace, TraceLevel

from utils.index import IndexList, StaticTuple


fn _argsort_cpu[
    *,
    ascending: Bool = True,
](
    indices: LayoutTensor[mut=True, address_space = AddressSpace.GENERIC, **_],
    input: LayoutTensor,
) raises:
    """
    Performs argsort on CPU.

    Parameters:
        ascending: Sort direction (True for ascending, False for descending).

    Args:
        indices: Output buffer to store sorted indices.
        input: Input buffer to sort.
    """

    @parameter
    fn fill_indices_iota[
        width: Int, rank: Int, alignment: Int = 1
    ](offset: IndexList[rank]):
        indices.ptr.store(offset[0], iota[indices.dtype, width](offset[0]))

    elementwise[fill_indices_iota, simdwidthof[indices.dtype](), target="cpu"](
        indices.size()
    )

    @parameter
    fn cmp_fn(a: Scalar[indices.dtype], b: Scalar[indices.dtype]) -> Bool:
        @parameter
        if ascending:
            return Bool(input[Int(a)] < input[Int(b)])
        else:
            return Bool(input[Int(a)] > input[Int(b)])

    sort[cmp_fn](
        Span[
            Scalar[indices.dtype],
            indices.origin,
        ](ptr=indices.ptr, length=indices.size())
    )


@always_inline
fn _sentinel_val[dtype: DType, ascending: Bool]() -> Scalar[dtype]:
    """
    Returns a sentinel value based on sort direction.

    Parameters:
        dtype: Data type of the sentinel value.
        ascending: Sort direction.

    Returns:
        MAX_FINITE for ascending sort, MIN_FINITE for descending sort.
    """

    @parameter
    if ascending:
        return Scalar[dtype].MAX_FINITE
    else:
        return Scalar[dtype].MIN_FINITE


fn _argsort_gpu_impl[
    *,
    ascending: Bool = True,
](
    indices: LayoutTensor[mut=True, **_],
    input: LayoutTensor,
    ctx: DeviceContext,
) raises:
    """
    Implements GPU argsort using bitonic sort algorithm.

    Parameters:
        ascending: Sort direction (True for ascending, False for descending).

    Args:
        indices: Output buffer to store sorted indices.
        input: Input buffer to sort.
        ctx: Device context for GPU execution.
    """
    # Create a device buffer to store a copy of the input data
    var n = indices.size()

    debug_assert(n.is_power_of_two(), "n must be a power of two")

    # Define block size for GPU kernel execution
    alias BLOCK_SIZE = 256

    # Bitonic sort algorithm implementation
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK_SIZE)
    )
    fn bitonic_sort_step(
        indices: LayoutTensor[mut=True, indices.dtype, indices.layout],
        input: LayoutTensor[mut=True, input.dtype, input.layout],
        n: Int,
        step: Int,
        stage: Int,
    ):
        """
        GPU kernel for one step of the bitonic sort algorithm.

        Args:
            indices: Buffer containing indices to sort.
            input: Buffer containing values to sort by.
            n: Total length of the buffers (padded).
            step: Current step size in the bitonic sequence.
            stage: Current stage of the bitonic sort.
        """
        var i = global_idx.x

        if i >= n:
            return

        var partner = i ^ step

        if partner > i and partner < n:
            var cmp_val: Bool

            @parameter
            if ascending:
                cmp_val = Bool(input[i] > input[partner])
            else:
                cmp_val = Bool(input[i] < input[partner])

            # Determine if we are in ascending or descending part of bitonic merge.
            var bitonic_merge_direction = (i & stage) == 0

            if cmp_val == bitonic_merge_direction:
                swap(input[i], input[partner])
                swap(indices[i], indices[partner])

    var k = 2
    # Iterate through increasing sequence lengths
    while k <= n:
        var j = k // 2
        while j > 0:
            # Launch GPU kernel for each stage of the bitonic sort
            ctx.enqueue_function[bitonic_sort_step](
                indices,
                input,
                n,
                j,
                k,
                block_dim=BLOCK_SIZE,
                grid_dim=ceildiv(n, BLOCK_SIZE),
            )
            j //= 2
        k *= 2


fn _argsort_gpu[
    *,
    ascending: Bool = True,
](
    indices: LayoutTensor[mut=True, **_],
    input: LayoutTensor,
    ctx: DeviceContext,
) raises:
    """
    Performs argsort on GPU with padding to power-of-two size if needed.

    Parameters:
        ascending: Sort direction (True for ascending, False for descending).

    Args:
        indices: Output buffer to store sorted indices.
        input: Input buffer to sort.
        ctx: Device context for GPU execution.
    """
    # Create a device buffer to store a copy of the input data
    var n = indices.size()

    if n.is_power_of_two():
        # Initialize indices with iota.
        @parameter
        @__copy_capture(indices)
        fn fill_indices_iota_no_padding[
            width: Int, rank: Int, alignment: Int = 1
        ](offset: IndexList[rank]):
            indices.ptr.store(offset[0], iota[indices.dtype, width](offset[0]))

        elementwise[
            fill_indices_iota_no_padding,
            simd_width = simdwidthof[
                indices.dtype, target = get_gpu_target()
            ](),
            target="gpu",
        ](n, ctx)

        return _argsort_gpu_impl[ascending=ascending](indices, input, ctx)

    var pow_2_length = next_power_of_two(n)

    # Else we need to pad the input and indices with sentinel values.

    var padded_input_buffer = ctx.enqueue_create_buffer[input.dtype](
        pow_2_length
    )
    var padded_input = LayoutTensor[
        mut=True, input.dtype, Layout.row_major(UNKNOWN_VALUE)
    ](
        padded_input_buffer,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            IndexList[1](pow_2_length)
        ),
    )

    var padded_indices_buffer = ctx.enqueue_create_buffer[indices.dtype](
        pow_2_length
    )
    var padded_indices = LayoutTensor[
        mut=True, indices.dtype, Layout.row_major(UNKNOWN_VALUE)
    ](
        padded_indices_buffer,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            IndexList[1](pow_2_length)
        ),
    )

    # Initialize indices with sequential values and copy input data to device
    @parameter
    @__copy_capture(padded_indices, padded_input, input, indices, n)
    fn fill_indices_iota[
        width: Int, rank: Int, alignment: Int = 1
    ](offset: IndexList[rank]):
        var i = offset[0]
        if i < n:
            padded_indices.ptr.store(i, iota[padded_indices.dtype, width](i))
            padded_input.ptr.store[alignment = padded_input.alignment](
                i, input.ptr.load[width=width](i)
            )
            return

        # otherwise we pad with a sentinel value and the max/min value for the type.
        padded_indices.ptr.store(
            i, SIMD[padded_indices.dtype, width](UNKNOWN_VALUE)
        )
        padded_input.ptr.store(
            i,
            SIMD[padded_input.dtype, width](
                _sentinel_val[padded_input.dtype, ascending]()
            ),
        )

    # we want to fill one element at a time to handle the case where n is not a
    # power of 2, so we set the simdwidth to be 1.
    elementwise[fill_indices_iota, simd_width=1, target="gpu"](
        pow_2_length, ctx
    )

    # Run the argsort implementation with the padded input and indices.
    _argsort_gpu_impl[ascending=ascending](padded_indices, padded_input, ctx)

    # Extract the unpadded indices from the padded indices.
    @parameter
    @__copy_capture(padded_indices, indices)
    fn extract_indices[
        width: Int, rank: Int, alignment: Int = 1
    ](offset: IndexList[rank]):
        indices.ptr.store(
            offset[0], padded_indices.ptr.load[width=width](offset[0])
        )

    # Extract the unpadded indices from the padded indices.
    elementwise[
        extract_indices,
        simd_width = simdwidthof[indices.dtype, target = get_gpu_target()](),
        target="gpu",
    ](n, ctx)

    # Free the temporary input buffer
    _ = padded_input_buffer^
    _ = padded_indices_buffer^


fn _validate_argsort(input: LayoutTensor, output: LayoutTensor) raises:
    """
    Validates input and output buffers for argsort operation.

    Args:
        input: Buffer containing values to sort.
        output: Buffer to store sorted indices.

    Raises:
        Error if buffers don't meet requirements for argsort.
    """

    @parameter
    if output.rank != 1:
        raise "output must be a 1D tensor"

    @parameter
    if input.rank != 1:
        raise "input must be a 1D tensor"

    @parameter
    if not output.dtype.is_integral():
        raise "output must be an integer type"

    if output.size() != input.size():
        raise "output and input must have the same length"


fn argsort[
    *,
    ascending: Bool = True,
    target: StaticString = "cpu",
](
    output: LayoutTensor[mut=True, **_], input: LayoutTensor, ctx: DeviceContext
) raises:
    """
    Performs argsort on input buffer, storing indices in output buffer.

    Parameters:
        ascending: Sort direction (True for ascending, False for descending).
        target: Target device ("cpu" or "gpu").

    Args:
        output: Buffer to store sorted indices.
        input: Buffer containing values to sort.
        ctx: Device context for execution.
    """
    with Trace[TraceLevel.OP, target=target]("argsort"):
        _validate_argsort(input, output)

        @parameter
        if is_cpu[target]():
            return _argsort_cpu[ascending=ascending](
                output.address_space_cast[AddressSpace.GENERIC](), input
            )
        else:
            return _argsort_gpu[ascending=ascending](output, input, ctx)


fn argsort[
    ascending: Bool = True
](output: LayoutTensor[mut=True, **_], input: LayoutTensor) raises:
    """
    CPU-only version of argsort.

    Parameters:
        ascending: Sort direction (True for ascending, False for descending).

    Args:
        output: Buffer to store sorted indices.
        input: Buffer containing values to sort.
    """
    with Trace[TraceLevel.OP]("argsort"):
        _validate_argsort(input, output)
        _argsort_cpu[ascending=ascending](
            output.address_space_cast[AddressSpace.GENERIC](), input
        )
