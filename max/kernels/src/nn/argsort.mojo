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


from collections import OptionalReg
from math import ceildiv, iota
from sys.info import simd_width_of

from algorithm import elementwise
from bit import next_power_of_two
from gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx
from gpu.host import DeviceContext, get_gpu_target
from gpu.host.info import is_cpu
from layout._coord import Coord, Idx
from layout._layout import row_major
from layout._tile_tensor import TileTensor
from runtime.tracing import Trace, TraceLevel, get_safe_task_id

from utils.index import IndexList, StaticTuple


fn _argsort_cpu[
    *,
    ascending: Bool = True,
](
    indices: TileTensor[mut=True, address_space = AddressSpace.GENERIC, ...],
    input: TileTensor,
) raises:
    """
    Performs argsort on CPU.

    Parameters:
        ascending: Sort direction (True for ascending, False for descending).

    Args:
        indices: Output buffer to store sorted indices.
        input: Input buffer to sort.
    """
    comptime assert input.flat_rank == 1

    @parameter
    fn fill_indices_iota[
        width: Int, rank: Int, alignment: Int = 1
    ](offset: IndexList[rank]):
        indices.ptr.store(
            offset[0],
            iota[indices.dtype, width](Scalar[indices.dtype](offset[0])),
        )

    elementwise[
        fill_indices_iota, simd_width_of[indices.dtype](), target="cpu"
    ](indices.numel())

    @parameter
    fn cmp_fn(a: Scalar[indices.dtype], b: Scalar[indices.dtype]) -> Bool:
        comptime if ascending:
            return input[a] < input[b]
        else:
            return input[a] > input[b]

    sort[cmp_fn](
        Span[
            Scalar[indices.dtype],
            indices.origin,
        ](ptr=indices.ptr, length=indices.numel())
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

    comptime if ascending:
        return Scalar[dtype].MAX_FINITE
    else:
        return Scalar[dtype].MIN_FINITE


fn _argsort_gpu_impl[
    *,
    ascending: Bool = True,
](
    indices: TileTensor[mut=True, ...],
    input: TileTensor[mut=True, ...],
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
    comptime assert input.flat_rank == 1
    comptime assert indices.flat_rank == 1
    # Create a device buffer to store a copy of the input data
    var n = indices.numel()

    debug_assert(n.is_power_of_two(), "n must be a power of two")

    # Define block size for GPU kernel execution
    comptime BLOCK_SIZE = 256

    # Bitonic sort algorithm implementation
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK_SIZE)
    )
    fn bitonic_sort_step(
        indices: TileTensor[indices.dtype, indices.LayoutType, indices.origin],
        input: TileTensor[input.dtype, input.LayoutType, input.origin],
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
        comptime assert input.flat_rank == 1
        comptime assert indices.flat_rank == 1
        var i = global_idx.x

        if i >= UInt(n):
            return

        var partner = i ^ UInt(step)

        if partner > i and partner < UInt(n):
            var cmp_val: Bool

            comptime if ascending:
                cmp_val = input[i] > input[partner]
            else:
                cmp_val = input[i] < input[partner]

            # Determine if we are in ascending or descending part of bitonic merge.
            var bitonic_merge_direction = (i & UInt(stage)) == 0

            if cmp_val == bitonic_merge_direction:
                swap(input[i], input[partner])
                swap(indices[i], indices[partner])

    var k = 2
    # Iterate through increasing sequence lengths
    while k <= n:
        var j = k // 2
        while j > 0:
            # Launch GPU kernel for each stage of the bitonic sort
            comptime kernel = bitonic_sort_step
            ctx.enqueue_function[kernel, kernel](
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
    indices: TileTensor[mut=True, ...],
    input: TileTensor[mut=True, ...],
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
    comptime assert indices.flat_rank == 1
    comptime assert input.flat_rank == 1
    # Create a device buffer to store a copy of the input data
    var n = indices.numel()

    if n.is_power_of_two():
        # Initialize indices with iota.
        @parameter
        @__copy_capture(indices)
        fn fill_indices_iota_no_padding[
            width: Int, rank: Int, alignment: Int = 1
        ](offset: IndexList[rank]):
            indices.ptr.store(
                offset[0],
                iota[indices.dtype, width](Scalar[indices.dtype](offset[0])),
            )

        elementwise[
            fill_indices_iota_no_padding,
            simd_width = simd_width_of[
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
    var padded_input = TileTensor(
        padded_input_buffer, row_major(Idx(pow_2_length))
    )

    var padded_indices_buffer = ctx.enqueue_create_buffer[indices.dtype](
        pow_2_length
    )
    var padded_indices = TileTensor(
        padded_indices_buffer,
        row_major(Idx(pow_2_length)),
    )

    # Initialize indices with sequential values and copy input data to device
    @parameter
    @__copy_capture(padded_indices, padded_input, input, indices, n)
    fn fill_indices_iota[
        width: Int, rank: Int, alignment: Int = 1
    ](offset: IndexList[rank]):
        var i = offset[0]
        if i < n:
            padded_indices.ptr.store(
                i, iota[padded_indices.dtype, width](Scalar[indices.dtype](i))
            )
            padded_input.ptr.store[
                alignment = simd_width_of[padded_input.dtype]()
            ](i, input.ptr.load[width=width](i))
            return

        # otherwise we pad with a sentinel value and the max/min value for the type.
        comptime UNKNOWN_VALUE = -1
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
        simd_width = simd_width_of[indices.dtype, target = get_gpu_target()](),
        target="gpu",
    ](n, ctx)

    # Free the temporary input buffer
    _ = padded_input_buffer^
    _ = padded_indices_buffer^


fn _validate_argsort(input: TileTensor, output: TileTensor) raises:
    """
    Validates input and output buffers for argsort operation.

    Args:
        input: Buffer containing values to sort.
        output: Buffer to store sorted indices.

    Raises:
        Error if buffers don't meet requirements for argsort.
    """

    if output.numel() != input.numel():
        raise "output and input must have the same length"


fn argsort[
    *,
    ascending: Bool = True,
    target: StaticString = "cpu",
](
    output: TileTensor[mut=True, address_space = AddressSpace.GENERIC, ...],
    input: TileTensor[mut=True, ...],
    ctx: DeviceContext,
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
    comptime assert input.flat_rank == 1
    comptime assert output.flat_rank == 1
    with Trace[TraceLevel.OP, target=target](
        "argsort",
        task_id=get_safe_task_id(ctx),
    ):
        _validate_argsort(input, output)

        comptime if is_cpu[target]():
            return _argsort_cpu[ascending=ascending](output, input)
        else:
            return _argsort_gpu[ascending=ascending](output, input, ctx)


fn argsort[
    ascending: Bool = True
](
    output: TileTensor[mut=True, address_space = AddressSpace.GENERIC, ...],
    input: TileTensor,
) raises:
    """
    CPU-only version of argsort.

    Parameters:
        ascending: Sort direction (True for ascending, False for descending).

    Args:
        output: Buffer to store sorted indices.
        input: Buffer containing values to sort.
    """
    comptime assert input.flat_rank == 1
    comptime assert output.flat_rank == 1
    comptime assert output.dtype.is_integral()
    with Trace[TraceLevel.OP]("argsort"):
        _validate_argsort(input, output)
        _argsort_cpu[ascending=ascending](output, input)
